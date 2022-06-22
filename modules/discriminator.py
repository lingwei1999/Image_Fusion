from torch import nn, Tensor


class Discriminator(nn.Module):
    """
    Use to discriminate fused images and source images.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, (3, 3), (2, 2), 1),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, (3, 3), (2, 2), 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, (3, 3), (2, 2), 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True)
            ),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 128, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
