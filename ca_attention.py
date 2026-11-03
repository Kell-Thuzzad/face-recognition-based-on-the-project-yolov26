import torch
import torch.nn as nn
import torch.nn.functional as F

class CAAttention(nn.Module):
    def __init__(self, in_channels):
        super(CAAttention, self).__init__()
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Conv2d(2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # 通道注意力
        avg_pool = self.pool(x).view(batch_size, C)
        attn = self.fc(avg_pool).view(batch_size, C, 1, 1)
        x = x * attn

        # 空间注意力
        h_pool = F.adaptive_avg_pool2d(x, (H, 1))
        w_pool = F.adaptive_avg_pool2d(x, (1, W))

        h_pool = h_pool.mean(dim=1, keepdim=True)
        w_pool = w_pool.mean(dim=1, keepdim=True)

        h_pool = F.interpolate(h_pool, size=(H, W), mode='bilinear', align_corners=False)
        w_pool = F.interpolate(w_pool, size=(H, W), mode='bilinear', align_corners=False)

        combined = torch.cat([h_pool, w_pool], dim=1)
        out = self.conv(combined)

        out = torch.sigmoid(out)
        return out * x
