import torch
import torch.nn as nn
from torchsummary import summary


def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, stride = 1, padding = 2, dilation = 2),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 2, dilation = 2),
        nn.ReLU()
    )


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_down1 = unet_block(3, 32)
        self.block_down2 = unet_block(32, 64)
        self.block_down3 = unet_block(64, 128)
        self.block_down4 = unet_block(128, 256)
        self.block_neck = unet_block(256, 512)
        self.block_up1 = unet_block(512 + 256, 256)
        self.block_up2 = unet_block(256 + 128, 128)
        self.block_up3 = unet_block(128 + 64, 64)
        self.block_up4 = unet_block(64 + 32, 32)
        self.conv_cls = nn.Conv2d(32, self.n_classes, 1) # -> (B, n_class, H, W)
    
    def forward(self, x):
        # (B, C, H, W)
        x1 = self.block_down1(x)
        x = self.downsample(x1)
        x2 = self.block_down2(x)
        x = self.downsample(x2)
        x3 = self.block_down3(x)
        x = self.downsample(x3)
        x4 = self.block_down4(x)
        x5 = self.downsample(x4)
        print(x4.size())

        x = self.block_neck(x5)
        print(self.upsample(x).size())
        x = torch.cat([x4, self.upsample(x)], dim = 1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim = 1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim = 1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim = 1)
        x = self.block_up4(x)

        x = self.conv_cls(x)
        return x
