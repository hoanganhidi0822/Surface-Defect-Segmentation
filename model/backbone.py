import torch
import timm
import torch,pdb
import torchvision
import torch.nn.modules
from torchsummary import summary



backbone = timm.create_model("resnet101", pretrained=True, features_only=True)
x = torch.rand(2, 3, 480, 640)
features = backbone(x)
for feature in features:
  print(feature.shape)
