import timm
import torch
import torch.nn as nn

def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, stride = 1, padding = 2, dilation = 2),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 2, dilation = 2),
        nn.ReLU()
    )

class ResUnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = timm.create_model("resnet101", pretrained=True, features_only=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        
        self.block_neck = unet_block(2048, 1024)
        self.block_up1 =  unet_block(1024 + 1024, 512)
        self.block_up2 =  unet_block(512 + 512, 256)
        self.block_up3 =  unet_block(256 + 256, 128)
        self.block_up4 =  unet_block(128 + 64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)
        #print(x1.shape)
        #print(x3.size())
        

        x = self.block_neck(x5) # 
        #print(x.size())
        #print(self.upsample(x).size())
        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)
        x = self.conv_cls(x) #size/2
        x = self.upsample(x)
        return x