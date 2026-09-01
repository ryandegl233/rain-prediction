import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CA(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CA, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle_channel = input_channels // reduction_ratio
        if middle_channel <= 0:
            middle_channel = input_channels
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        out = torch.clamp(out, min=-50, max=50) # 防止 Sigmoid 输入过大
        sig_out = torch.sigmoid(out.float()).type_as(x) # FP32 安全计算
        
        scale = x * sig_out.unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        sig_out = torch.sigmoid(out.float()).type_as(x)
        
        scale = x * sig_out
        return scale

class CASA(nn.Module):
    def __init__(self, in_channel):
        super(CASA, self).__init__()
        self.ca = CA(in_channel)
        self.sa = SA()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class SCA(nn.Module):
    """
    SCA (Spatial Coordinate Attention)
    --------------------------------------------------
    Logic:
    1. Coordinate Pooling: 分别沿 H 和 W 方向进行平均池化。
    2. Concatenation: 拼接 H 和 W 特征，建立空间关联。
    3. Shared Transform: 1x1 Conv -> LayerNorm -> GELU 。
    4. Split & Excite: 分割回 H, W 特征 -> Sigmoid -> 加权原特征。
    
    该模块专注于捕捉空间维度的注意力信息，
    与负责通道筛选的 CASF 形成 [空间 + 通道] 互补。
    """
    def __init__(self, inp, reduction=32):
        super(SCA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # (B, C, 1, W)

        mip = max(8, inp // reduction)

        # 1x1 Conv 用于降维和特征交互
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        

        self.ln = nn.LayerNorm(mip) 
        self.act = nn.GELU()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: (B, C, H, W)
        identity = x
        n, c, h, w = x.size()
        
        # 1. Coordinate Pooling
        x_h = self.pool_h(x) # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W) -> (B, C, W, 1)

        # 2. Concat
        y = torch.cat([x_h, x_w], dim=2) # (B, C, H+W, 1)
        
        # 3. Shared Transform (Modernized)
        y = self.conv1(y) # (B, mip, H+W, 1)
        
        # LN expects last dim to be channel, so we permute
        y = y.permute(0, 2, 3, 1) # (B, H+W, 1, mip)
        y = self.ln(y)
        y = self.act(y)
        y = y.permute(0, 3, 1, 2) # (B, mip, H+W, 1)
        
        # 4. Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # (B, mip, 1, W)

        # 5. Excite
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w

        return out
