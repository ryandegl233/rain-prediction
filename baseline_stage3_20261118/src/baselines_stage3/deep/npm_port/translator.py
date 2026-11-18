from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, batch_norm: bool) -> None:
        super().__init__()
        norm = nn.BatchNorm2d if batch_norm else nn.InstanceNorm2d
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            norm(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class SatelliteToRain(nn.Module):
    def __init__(self, channels: int, width: int, downsamples: int, residual_blocks: int, batch_norm: bool) -> None:
        super().__init__()
        norm = nn.BatchNorm2d if batch_norm else nn.InstanceNorm2d
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, width, 7),
            norm(width),
            nn.ReLU(inplace=True),
        ]
        hidden = width
        for _ in range(downsamples):
            layers.extend([nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), norm(hidden * 2), nn.ReLU(inplace=True)])
            hidden *= 2
        layers.extend(ResidualBlock(hidden, batch_norm) for _ in range(residual_blocks))
        for _ in range(downsamples):
            layers.extend(
                [
                    nn.ConvTranspose2d(hidden, hidden // 2, 3, stride=2, padding=1, output_padding=1),
                    norm(hidden // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            hidden //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(hidden, 1, 7)])
        self.model = nn.Sequential(*layers)
        self.act = nn.Identity()
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight, 0.0, 0.02)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.model(x))
