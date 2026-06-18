import torch
import torch.nn as nn


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
        resnet_blocks_number = 5,
    ):
        super().__init__()

        self.resnet_blocks_number = resnet_blocks_number

        self.first_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = inner_channels[0],
            kernel_size = 3,
            padding = 1
        )

        self.down1 = self.down_step(inner_channels[0], inner_channels[1])
        self.down2 = self.down_step(inner_channels[1], inner_channels[2])
        self.down3 = self.down_step(inner_channels[2], inner_channels[3])

        self.up3 = self.up_step(inner_channels[3] + inner_channels[2], inner_channels[2])
        self.up2 = self.up_step(inner_channels[2] + inner_channels[1], inner_channels[1])
        self.up1 = self.up_step(inner_channels[1] + inner_channels[0], inner_channels[0])

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
            *[ResBlock(out_channels) for _ in range(self.resnet_blocks_number)]
        )
    
    def up_step(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 2,
                stride = 2
            ),
            *[ResBlock(out_channels) for _ in range(self.resnet_blocks_number)]
        )
    
    def forward(self, x):
        source_x = x

        x0 = self.first_conv(x)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        y = self.up3(torch.cat([x3, x2], dim=1))
        y = self.up2(torch.cat([y, x1], dim=1))
        y = self.up1(torch.cat([y, x0], dim=1))

        return source_x + self.last_conv(y)