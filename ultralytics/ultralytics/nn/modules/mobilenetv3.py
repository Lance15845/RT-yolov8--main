import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class HSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3, 0, 6) / 6


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = HSigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, 1, 1, 0),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MobileNetV3Block(nn.Module):
    """
    MobileNetV3 inverted residual block
    args:
        c1: input channels
        c2: output channels
        k: kernel size
        s: stride
        exp: expansion channels
        se: use se module
        hs: use h-swish, else relu
    """
    def __init__(self, c1, c2, k=3, s=1, exp=16, se=False, hs=False):
        super().__init__()
        self.use_res = (s == 1 and c1 == c2)
        act = HSwish() if hs else nn.ReLU(inplace=True)

        layers = []

        # expand
        if exp != c1:
            layers.append(nn.Conv2d(c1, exp, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(exp))
            layers.append(act)

        # depthwise
        layers.append(nn.Conv2d(exp, exp, k, s, k // 2, groups=exp, bias=False))
        layers.append(nn.BatchNorm2d(exp))
        layers.append(act)

        # se
        if se:
            layers.append(SEModule(exp))

        # project
        layers.append(nn.Conv2d(exp, c2, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(c2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        if self.use_res:
            return x + y
        return y