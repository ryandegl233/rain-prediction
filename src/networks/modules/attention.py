import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## 混合注意力机制
def mix_attention(
    q,
    k,
    v,
    attn_mask=None,
    softmax_scale=None,
    dropout=0.0,
):
    o = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout,
        scale=softmax_scale,
        is_causal=False,
    )
    return o


class MultiHeadAttention(nn.Module):
    """标准多头自注意力机制"""

    def __init__(self, dim, num_heads=8, dropout=0.0, softmax_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = softmax_scale or self.head_dim**-0.5

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (bsz, num_heads, seq_len, head_dim)

        attn_output = mix_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            softmax_scale=self.scale,
            dropout=self.attn_dropout.p if self.training else 0.0,
        )

        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.out_proj(attn_output)


class AttentionSink(nn.Module):
    """真正的 Attention Sink 实现

    参考论文: "Efficient Streaming Language Models with Attention Sinks"
    核心思想: 保留初始的几个 sink tokens 来稳定注意力分布
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, num_sink_tokens=4, window_size=2048):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size

        assert dim % num_heads == 0

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

        # Sink token embeddings
        self.sink_tokens = nn.Parameter(torch.randn(1, num_sink_tokens, dim))

    def forward(self, x, use_cache=False, past_key_values=None):
        bsz, seq_len, _ = x.size()

        # 添加 sink tokens
        sink_tokens = self.sink_tokens.expand(bsz, -1, -1)
        x_with_sinks = torch.cat([sink_tokens, x], dim=1)

        # 生成 QKV
        qkv = self.qkv_proj(x_with_sinks).reshape(
            bsz, seq_len + self.num_sink_tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # 处理缓存（用于流式处理）
        if past_key_values is not None:
            past_k, past_v = past_key_values
            # 保留 sink tokens 和最近的 window_size 个 tokens
            if past_k.size(-2) > self.num_sink_tokens + self.window_size:
                # 保留 sink tokens 和最后的 window_size 个 tokens
                keep_indices = list(range(self.num_sink_tokens)) + list(
                    range(past_k.size(-2) - self.window_size, past_k.size(-2))
                )
                past_k = past_k[:, :, keep_indices, :]
                past_v = past_v[:, :, keep_indices, :]

            k = torch.cat([past_k, k[:, :, self.num_sink_tokens :, :]], dim=-2)
            v = torch.cat([past_v, v[:, :, self.num_sink_tokens :, :]], dim=-2)

        # 注意力计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0, scale=self.scale
        )

        # 移除 sink tokens 的输出（只返回原始序列的输出）
        attn_output = attn_output[:, :, self.num_sink_tokens :, :]
        attn_output = attn_output.reshape(bsz, seq_len, -1)

        output = self.out_proj(attn_output)

        if use_cache:
            # 返回当前的 k, v 用于下次缓存
            current_k = k[:, :, self.num_sink_tokens :, :]
            current_v = v[:, :, self.num_sink_tokens :, :]
            return output, (current_k, current_v)

        return output


class CrossAttention(nn.Module):
    """交叉注意力 - 用于雨量站信息与图像特征的融合"""

    def __init__(self, query_dim, key_dim, value_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(value_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (bsz, query_len, query_dim) - 例如图像位置的查询
            key: (bsz, key_len, key_dim) - 例如雨量站的位置特征
            value: (bsz, value_len, value_dim) - 例如雨量站的数值特征
        """
        bsz, query_len, _ = query.size()
        key_len = key.size(1)

        q = (
            self.q_proj(query)
            .reshape(bsz, query_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k_proj(key).reshape(bsz, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(bsz, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, query_len, -1)
        return self.out_proj(attn_output)


class SpatialAttention(nn.Module):
    """空间注意力 - 专门用于地理空间数据"""

    def __init__(self, dim, num_heads=8, dropout=0.0, max_distance=None):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.max_distance = max_distance

        # 位置编码
        self.pos_encoding = nn.Sequential(
            nn.Linear(2, dim // 4),  # 输入经纬度
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, features, coordinates, attn_mask=None):
        """
        Args:
            features: (bsz, num_points, dim) - 特征
            coordinates: (bsz, num_points, 2) - 经纬度坐标
        """
        # 添加位置编码
        pos_encoding = self.pos_encoding(coordinates)
        features_with_pos = features + pos_encoding

        # 基于距离的注意力掩码
        if self.max_distance is not None:
            distance_mask = self.create_distance_mask(coordinates)
            if attn_mask is not None:
                attn_mask = attn_mask & distance_mask
            else:
                attn_mask = distance_mask

        return self.attention(features_with_pos, attn_mask)

    def create_distance_mask(self, coordinates):
        """创建基于地理距离的注意力掩码"""
        bsz, num_points, _ = coordinates.size()

        # 计算所有点对之间的距离
        coord_i = coordinates.unsqueeze(2)  # (bsz, num_points, 1, 2)
        coord_j = coordinates.unsqueeze(1)  # (bsz, 1, num_points, 2)

        # 简化的欧几里得距离
        distances = torch.norm(coord_i - coord_j, dim=-1)  # (bsz, num_points, num_points)

        # 创建掩码
        mask = distances <= self.max_distance
        return mask