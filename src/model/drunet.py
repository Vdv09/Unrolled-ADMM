import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3, 
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3, 
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        residual = self.relu(self.conv1(x))
        residual = self.conv2(residual)

        return residual + x


class DRUNet(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        inner_channels,  # expect 4 channels numbers
        resnet_blocks_number = 4,
    ):
        super().__init__()

        self.first_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = inner_channels[0],
            kernel_size = 3,
            padding=1
        )

        self.resnet_blocks_number = resnet_blocks_number

        self.down1 = self.down_step(inner_channels[0], inner_channels[1])
        self.down2 = self.down_step(inner_channels[1], inner_channels[2])
        self.down3 = self.down_step(inner_channels[2], inner_channels[3])

        self.up3_upsample = nn.ConvTranspose2d(inner_channels[3], inner_channels[2], 2, stride=2)
        self.up3_concatenate = nn.Conv2d(inner_channels[2] * 2, inner_channels[2], 1)
        self.up3_resnet = nn.Sequential(*[ResBlock(inner_channels[2]) for _ in range(self.resnet_blocks_number)])

        self.up2_upsample = nn.ConvTranspose2d(inner_channels[2], inner_channels[1], 2, stride=2)
        self.up2_concatenate = nn.Conv2d(inner_channels[1] * 2, inner_channels[1], 1)
        self.up2_resnet = nn.Sequential(*[ResBlock(inner_channels[1]) for _ in range(self.resnet_blocks_number)])

        self.up1_upsample = nn.ConvTranspose2d(inner_channels[1], inner_channels[0], 2, stride=2)
        self.up1_concatenate = nn.Conv2d(inner_channels[0] * 2, inner_channels[0], 1)
        self.up1_resnet = nn.Sequential(*[ResBlock(inner_channels[0]) for _ in range(self.resnet_blocks_number)])

        self.last_conv = nn.Conv2d(
            in_channels = inner_channels[0], 
            out_channels = out_channels,
            kernel_size = 3,
            padding=1
        )

    def down_step(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            *[ResBlock(out_channels) for _ in range(self.resnet_blocks_number)],
        )

    def up_step(self, deep, skip, upsample, concatenate, resnet):
        y = upsample(deep)

        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        return resnet(concatenate(torch.cat([y, skip], dim=1)))

    def forward(self, x):
        x0 = self.first_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        y = self.up_step(x3, x2, self.up3_upsample, self.up3_concatenate, self.up3_resnet)
        y = self.up_step(y, x1, self.up2_upsample, self.up2_concatenate, self.up2_resnet)
        y = self.up_step(y, x0, self.up1_upsample, self.up1_concatenate, self.up1_resnet)

        return x + self.last_conv(y)
