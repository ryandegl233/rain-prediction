# -*- coding: utf-8 -*-
"""
SOTA Architecture: Time-Conditioned U-ViT with Eulerian-Lagrangian Advection
【OOM 显存优化版】
创新点:
1. Spatiotemporal Tubelet Compression (T=4 -> T=1)
2. DiT-style Deep Time Modulation (AdaLN-Zero) at multiple scales
3. Global Self-Attention Bottleneck with PyTorch Native FlashAttention (Scaled Dot Product)
4. Differentiable Physics Warping Layer (Lagrangian Advection)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 时间编码与深度注入模块 (DiT-Style)
# ==========================================

class SinusoidalTimeEmbedding(nn.Module):
    """连续时间步 \Delta t 的高频正弦映射"""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        frequencies = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32,
                                                                          device=t.device) / half_dim)
        args = t[:, None].float() * frequencies[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class TimeModulatedResBlock(nn.Module):
    """带时间注入的非线性残差块 (AdaLN调制)"""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # 将时间编码投影为 Scale(缩放) 和 Shift(平移)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # 深度时间注入
        time_scale, time_shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + time_scale) + time_shift

        h = self.conv2(F.silu(h))
        return x + h


# ==========================================
# 2. 空间注意力机制与物理形变模块
# ==========================================

class GlobalAttentionBottleneck(nn.Module):
    """用于捕捉大范围天气系统（如台风螺旋）的全局自注意力层 (已开启显存优化)"""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # 展平为序列，形状必须为 [B, heads, seq_len, head_dim] 以支持 SDPA
        q = q.view(B, self.heads, C // self.heads, H * W).transpose(-1, -2)
        k = k.view(B, self.heads, C // self.heads, H * W).transpose(-1, -2)
        v = v.view(B, self.heads, C // self.heads, H * W).transpose(-1, -2)

        # 🌟 核心优化：使用 PyTorch 原生 FlashAttention 接口，消除 OOM 瓶颈
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class NeuralAdvectionWarping(nn.Module):
    """可微流体力学拉格朗日平流层"""

    def __init__(self):
        super().__init__()

    def forward(self, last_frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = last_frame.shape
        device = last_frame.device

        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device),
                              indexing='ij')
        grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        flow_norm = flow.permute(0, 2, 3, 1).clone()
        flow_norm[..., 0] = flow_norm[..., 0] / (W / 2.0)
        flow_norm[..., 1] = flow_norm[..., 1] / (H / 2.0)

        warped_grid = grid + flow_norm
        return F.grid_sample(last_frame, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)


# ==========================================
# 3. Time-Conditioned U-ViT 主网络
# ==========================================

class TimeConditionedAdvectionTransformer(nn.Module):
    def __init__(self, radar_out_channels: int = 1, satellite_out_channels: int = 10, rain_out_channels: int = 1,
                 embed_dim: int = 256):
        super().__init__()
        self.r_c = radar_out_channels
        self.s_c = satellite_out_channels
        self.rain_c = rain_out_channels
        self.total_c = self.r_c + self.s_c + self.rain_c

        # 时间 prompt 编码器
        self.time_emb_dim = embed_dim
        self.time_encoder = SinusoidalTimeEmbedding(dim=self.time_emb_dim)

        # 1. 极致压缩 Stem (将 T=4 直接压平融合，保留 512x512 高分辨率)
        self.stem = nn.Conv2d(self.total_c * 4, 64, kernel_size=3, padding=1)

        # 2. U-Net 编码器 (逐级降维提取特征 + 时间注入，🌟已加深至 64x64 分辨率以拯救显存)
        self.down1 = TimeModulatedResBlock(64, self.time_emb_dim)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # -> 256x256
        self.down3 = TimeModulatedResBlock(128, self.time_emb_dim)
        self.down4 = nn.Conv2d(128, embed_dim, kernel_size=4, stride=2, padding=1)  # -> 128x128
        self.down5 = TimeModulatedResBlock(embed_dim, self.time_emb_dim)
        self.down6 = nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1)  # -> 64x64 (极大降低 Attention 内存)

        # 3. Transformer 瓶颈层 (结合全局 Attention 与深度时间注入)
        self.mid_block1 = TimeModulatedResBlock(embed_dim, self.time_emb_dim)
        self.mid_attn = GlobalAttentionBottleneck(embed_dim, heads=8)
        self.mid_block2 = TimeModulatedResBlock(embed_dim, self.time_emb_dim)

        # 4. U-Net 解码器 (逐级上采样 + 时间注入)
        self.up0 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1)  # -> 128x128
        self.up_block0 = TimeModulatedResBlock(embed_dim, self.time_emb_dim)
        self.up1 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1)  # -> 256x256
        self.up_block1 = TimeModulatedResBlock(128, self.time_emb_dim)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> 512x512
        self.up_block2 = TimeModulatedResBlock(64, self.time_emb_dim)

        # 5. 分支物理预测头
        self.norm_out = nn.GroupNorm(8, 64)
        self.flow_head = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.residual_head = nn.Conv2d(64, self.total_c, kernel_size=3, padding=1)

        self.warper = NeuralAdvectionWarping()

        # 零初始化（保证训练初期网络不乱动，平滑起步）
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, context_x: torch.Tensor, delta_t: torch.Tensor, **kwargs) -> dict:
        B, C, T, H, W = context_x.shape
        last_frame = context_x[:, :, -1, :, :]  # [B, C, H, W]

        # A. 生成当前时刻的时间 Prompt
        t_emb = self.time_encoder(delta_t)  # [B, embed_dim]

        # B. 时空维度折叠融合 (T=4 合并到 Channel 维度)
        x_flat = context_x.view(B, C * T, H, W)
        x = self.stem(x_flat)

        # C. 特征下采样与调制
        x = self.down1(x, t_emb)
        x = self.down2(x)
        x = self.down3(x, t_emb)
        x = self.down4(x)
        x = self.down5(x, t_emb)
        x = self.down6(x)

        # D. 全局 Transformer 建模大尺度环流
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # E. 特征上采样与解码
        x = self.up0(x)
        x = self.up_block0(x, t_emb)
        x = self.up1(x)
        x = self.up_block1(x, t_emb)
        x = self.up2(x)
        x = self.up_block2(x, t_emb)
        x = F.silu(self.norm_out(x))

        # F. Eulerian-Lagrangian 物理场预测
        flow = self.flow_head(x)  # 光流位移
        residual = self.residual_head(x)  # 对流残差

        # G. 物理形变引擎执行 5/10/15分钟 平流形变
        warped_frame = self.warper(last_frame, flow)
        final_prediction = warped_frame + residual

        # H. 对齐外部验证框架的数据格式
        return {
            "radar": final_prediction[:, :self.r_c].unsqueeze(2),
            "satellite": final_prediction[:, self.r_c: self.r_c + self.s_c].unsqueeze(2),
            "rain": final_prediction[:, self.r_c + self.s_c:].unsqueeze(2),
            "flow": flow,
            "residual": residual
        }