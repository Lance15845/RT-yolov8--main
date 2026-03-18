import torch.nn as nn


class MobileNetV2Block(nn.Module):
    """MobileNetV2 inverted residual block used as a lightweight backbone primitive."""

    def __init__(self, c1, c2, k=3, s=1, exp=16):
        super().__init__()
        hidden_dim = max(int(exp), c1)
        self.use_residual = s == 1 and c1 == c2

        layers = []
        if hidden_dim != c1:
            layers.extend(
                [
                    nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, k, s, k // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            return x + y
        return y
