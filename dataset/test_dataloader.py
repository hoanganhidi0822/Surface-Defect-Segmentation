import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2 # np.array -> torch.tensor
import os
from PIL import Image
from glob import glob
from dataloader import DataRoad
from torchvision import transforms
trainsize = 384
img_hight,img_width = 160, 320

def binary_mask_to_rgb(rgb_image, true_color=[31, 120, 180], false_color=[0, 0, 0]):
    
    rgb_image = np.array(rgb_image)

   
    binary_mask = (rgb_image == 1).all(axis=-1)

    h, w = rgb_image.shape
    final_rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    
    final_rgb_image[binary_mask] = true_color
    final_rgb_image[~binary_mask] = false_color

    return final_rgb_image
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
train_transform = A.Compose([
    A.Resize(width=160, height=80),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    #A.Cutout(num_holes=5, max_h_size=15, max_w_size=15, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])


train_dataset = DataRoad(train_transform)
print(len(train_dataset))


image, mask = train_dataset.__getitem__(10200)

#mask = binary_mask_to_rgb(mask)
#plt.subplot(1, 2, 1)
#plt.imshow(image)
# plt.subplot(1, 2, 2)
plt.imshow(unorm(image).permute(1, 2, 0))

plt.axis('off')

plt.show()