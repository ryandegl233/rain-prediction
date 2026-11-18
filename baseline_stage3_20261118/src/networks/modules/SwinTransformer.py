import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from src.networks.modules.Attention import CASA,SCA
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GRU(nn.Module):
    def __init__(self, channel, h_w, kernel_size=3, stride=1, padding=1):
        super(GRU, self).__init__()
        height, width = h_w
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(1, channel * 2, eps=1e-3),
            nn.GroupNorm(1, channel * 2, eps=1e-3),
        )
        """
        ----影响大批量数据加载时的稳定性---
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(1, channel),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(1, channel),
        )
        """
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, g, l):
        cat_1 = self.conv_1(torch.cat((g, l), dim=1))
        Z, R = torch.chunk(cat_1, 2, dim=1)
        Z = torch.sigmoid(Z.float()).type_as(g)
        R = torch.sigmoid(R.float()).type_as(g)

        H = Z * g + R * l

        return H


class CASF(nn.Module):
    """
    CASF (Channel-Aware Selective Fusion)
    --------------------------------------------------
    逻辑流程:
    1. Cat: [Local, Global] 拼接，获取联合上下文。
    2. Weights: GAP -> MLP -> Softmax 生成权重 a (Local) 和 b (Global)。
    3. Local Path: Out_L =  a * Conv(Local) 
    4. Global Path: Out_G =  b * Conv(Global) 
    5. Fusion: Out = Out_L + Out_G
    """
    def __init__(self, dim, reduction=8):
        super(CASF, self).__init__()
        mid_dim = max(dim // reduction, 32)
        
        # 1. Attention Generator
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, mid_dim, 1, bias=False),
            nn.LayerNorm([mid_dim, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim * 2, 1, bias=True) # 输出 2*dim 用于 Split
        )
        self.softmax = nn.Softmax(dim=1)
        
        # 2. Branch Transforms (1x1 Conv for Projection/Refinement)
        # 使用 1x1 卷积进行特征变换，保持高效
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.global_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, global_x, local_x):
        # global_x, local_x: [B, C, H, W]
        B, C, H, W = global_x.shape
        
        # --- Step 1: Concat (联合感知) ---
        cat_feat = torch.cat([local_x, global_x], dim=1) # [B, 2C, H, W]
        
        # --- Step 2: Weight Generation (生成权重) ---
        s = self.avg_pool(cat_feat) # [B, 2C, 1, 1]
        z = self.mlp(s)             # [B, 2C, 1, 1]
        
        # Reshape for Softmax over the 2 branches
        z = z.view(B, 2, C, 1, 1)
        weights = self.softmax(z)
        a = weights[:, 0] # Weight for Local [B, C, 1, 1]
        b = weights[:, 1] # Weight for Global [B, C, 1, 1]
        
        # --- Step 3: Local Path Processing  ---
        local_transformed = self.local_conv(local_x)
        local_weighted = local_transformed * a
        #branch_local = local_transformed + local_weighted
        
        # --- Step 4: Global Path Processing  ---
        global_transformed = self.global_conv(global_x)
        global_weighted = global_transformed * b
        #branch_global = global_transformed + global_weighted
        
        # --- Step 5: Final Fusion  ---
        out = local_weighted + global_weighted
        #out = branch_local + branch_global
        
        return out


class CA(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CA, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle_channel = input_channels // reduction_ratio
        if middle_channel < 10:
            middle_channel = input_channels
        self.MLP1 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )
        self.MLP2 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP1(avg_values) + self.MLP2(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = scale.permute(0, 2, 3, 1)
        return scale


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    """
    窗口注意力机制模块
    实现了在局部窗口内进行自注意力计算的机制，支持相对位置编码和窗口移动
    通过将输入特征图划分为多个窗口，计算每个窗口内的自注意力，从而捕捉局部特征
    该模块可以选择是否使用相对位置编码，以增强模型对空间关系的理解能力
    还可以选择是否进行窗口移动(shifted)，以实现跨窗口的信息交互
    主要参数包括输入维度(dim)、注意力头数(heads)、每个头的维度(head_dim)、窗口大小(window_size)等
    通过这些参数，可以灵活调整注意力机制的计算方式和效果
    """
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

        self.ca = CA(dim)

    def forward(self, x):
        if self.shifted:
            # 左上角移动
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        # 计算 Attention 分数
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        # 加入相对位置编码
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        # Mask 处理
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

       # ==================== ⚠️ 修改开始 ====================
        # 1. 数值截断 (你之前加的，保留)
        dots = torch.clamp(dots, min=-50.0, max=50.0)
        
        # 2. 【关键修改】强制转为 float32 进行 Softmax
        # 即使你在用 AMP (自动混合精度)，这里也必须用 float32，否则极易溢出
        attn = F.softmax(dots.float(), dim=-1)
        
        # 3. 再转回原来的精度 (比如 fp16) 以继续后面的计算
        attn = attn.type_as(dots)

        # 4. 兜底 (你之前加的，保留)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
        # ==================== ⚠️ 修改结束 ====================

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        out = self.ca(out)

        return out


class SwinBlock(nn.Module):
    """
    注意力块和mlp组合模块
    """
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    """
    下采样模块
    -通过线性变换将相邻像素块合并，从而减少空间分辨率并增加通道数
    -例如，输入特征图大小为 (B, C, H, W)，下采样因子为 2,则输出特征图大小为 (B, C * 4, H/2, W/2)
    -通过将每个 2x2 像素块展平并通过线性层映射到更高维度来实现
    """
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = torch.reshape(x, (b, c, new_h, self.downscaling_factor, new_w, self.downscaling_factor))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = torch.reshape(x, (b, c * (self.downscaling_factor ** 2), new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class PatchExpanding(nn.Module):
    """
    上采样模块
    -通过线性变换将特征图的空间分辨率增加，同时减少通道数
    """
    def __init__(self, in_channels, out_channels, upscaling_factor):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.linear = nn.Linear(in_channels // (upscaling_factor ** 2), out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h * self.upscaling_factor, w * self.upscaling_factor
        new_c = int(c // (self.upscaling_factor ** 2))
        x = torch.reshape(x, (b, new_c, self.upscaling_factor, self.upscaling_factor, h, w))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = torch.reshape(x, (b, new_c, new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, h_w):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                DoubleConv(hidden_dimension, hidden_dimension),
                #SCA(hidden_dimension),
                CASA(hidden_dimension),
                #CASF(dim=hidden_dimension),
                GRU(hidden_dimension, h_w=h_w)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block, cnn, casa, gru in self.layers:
            x = x.permute(0, 3, 1, 2)
            local_x = cnn(x)
            local_x = casa(local_x)
            x = x.permute(0, 2, 3, 1)
            global_x = regular_block(x)
            global_x = shifted_block(global_x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = gru(global_x, local_x)
        return out

class StageModule_up(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, h_w):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.in_channel = in_channels
        self.hidden_dimension = hidden_dimension

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension * 2, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension * 2, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                DoubleConv(hidden_dimension * 2, hidden_dimension * 2),
                #SCA(hidden_dimension * 2),
                CASA(hidden_dimension * 2),
                #CASF(dim=hidden_dimension * 2),
                GRU(hidden_dimension * 2, h_w=h_w)
            ]))

    def forward(self, x, x2):
        x = self.patch_partition(x)
        x2 = x2.permute(0, 2, 3, 1)
        x = torch.cat((x, x2), dim=-1)
        for regular_block, shifted_block, cnn, casa, gru in self.layers:
            x = x.permute(0, 3, 1, 2)
            local_x = cnn(x)
            local_x = casa(local_x)
            x = x.permute(0, 2, 3, 1)
            global_x = regular_block(x)
            global_x = shifted_block(global_x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = gru(global_x, local_x)
        return out


class StageModule_up_final(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, h_w):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                DoubleConv(hidden_dimension, hidden_dimension),
                #SCA(hidden_dimension),
                CASA(hidden_dimension),
                #CASF(dim=hidden_dimension),
                GRU(hidden_dimension, h_w=h_w)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block, cnn, casa, gru in self.layers:
            x = x.permute(0, 3, 1, 2)
            local_x = cnn(x)
            local_x = casa(local_x)
            x = x.permute(0, 2, 3, 1)
            global_x = regular_block(x)
            global_x = shifted_block(global_x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = gru(global_x, local_x)

        return out


