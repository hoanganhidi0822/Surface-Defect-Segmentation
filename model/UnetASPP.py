import torch
import torch.nn as nn

def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU()
    )

class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPBlock, self).__init__()
        dilations = [2, 3, 4, 5]

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        conv1x1 = self.conv1(x)
        conv3x3_1 = self.conv2(x)
        conv3x3_2 = self.conv3(x)
        conv3x3_3 = self.conv4(x)
        conv3x3_4 = self.conv5(x)

        global_avg_pool = self.global_avg_pool(x)
        global_avg_pooling = self.conv1x1_out(global_avg_pool)

        print("conv1x1 size:", conv1x1.size())
        print("conv3x3_1 size:", conv3x3_1.size())
        print("conv3x3_2 size:", conv3x3_2.size())
        print("conv3x3_3 size:", conv3x3_3.size())
        print("conv3x3_4 size:", conv3x3_4.size())
        print("global_avg_pool size:", global_avg_pooling.size())

        out = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4, global_avg_pooling.expand(-1, -1, conv1x1.size(2), conv1x1.size(3))], dim=1)


        return out

class UNetWithASPP(nn.Module):
    def __init__(self, n_classes=2):
        super(UNetWithASPP, self).__init__()
        self.n_classes = n_classes

        self.block_down1 = unet_block(3, 16)
        self.block_down2 = unet_block(16, 32)
        self.block_down3 = unet_block(32, 64)
        self.block_down4 = unet_block(64, 128)

        self.aspp = ASPPBlock(128, 256)

        self.block_up1 = unet_block(256 + 128, 128)
        self.block_up2 = unet_block(128 + 64, 64)
        self.block_up3 = unet_block(64 + 32, 32)
        self.block_up4 = unet_block(32 + 16, 16)

        self.conv_cls = nn.Conv2d(16, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.block_down1(x)
        x2 = self.block_down2(x1)
        x3 = self.block_down3(x2)
        x4 = self.block_down4(x3)

        x_aspp = self.aspp(x4)

        x = torch.cat([x4, x_aspp], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, x], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.block_up4(x)

        x = self.conv_cls(x)
        return x

# Example usage
model_with_aspp = UNetWithASPP(n_classes=2)
x = torch.rand(4, 3, 80, 160)  # batch size, channels, height, width
print("Input shape =", x.shape)

y_with_aspp = model_with_aspp(x).squeeze()
print("Output shape with ASPP =", y_with_aspp.shape)