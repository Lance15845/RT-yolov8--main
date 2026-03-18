import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class GhostConv(nn.Module):
    """
    Ghost Convolution: 用少量主卷积 + cheap operation 生成更多特征
    """
    def __init__(self, c1, c2, k=1, s=1, ratio=2, dwk=3, act=True):
        super().__init__()
        # 主分支输出通道
        c_ = int((c2 + ratio - 1) // ratio)
        # cheap 分支输出通道
        c_cheap = c2 - c_

        self.primary = Conv(c1, c_, k, s, act=act)
        self.cheap = Conv(c_, c_cheap, dwk, 1, g=c_, act=act) if c_cheap > 0 else nn.Identity()

    def forward(self, x):
        y = self.primary(x)
        y2 = self.cheap(y)
        return torch.cat((y, y2), 1)


class GhostBottleneck(nn.Module):
    """
    GhostNet Bottleneck（简化版）
    """
    def __init__(self, c1, c2, k=3, s=1, ratio=2):
        super().__init__()
        c_mid = c2 // 2

        self.ghost1 = GhostConv(c1, c_mid, k=1, s=1, ratio=ratio)
        self.dw = Conv(c_mid, c_mid, k, s, g=c_mid, act=False) if s == 2 else nn.Identity()
        self.ghost2 = GhostConv(c_mid, c2, k=1, s=1, ratio=ratio, act=False)

        self.shortcut = (
            nn.Identity()
            if (s == 1 and c1 == c2)
            else nn.Sequential(
                Conv(c1, c1, k=3, s=s, g=c1, act=False),
                Conv(c1, c2, k=1, s=1, act=False),
            )
        )

    def forward(self, x):
        y = self.ghost1(x)
        y = self.dw(y)
        y = self.ghost2(y)
        return y + self.shortcut(x)