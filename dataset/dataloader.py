import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from glob import glob
import os

img_width, img_height = 640, 480

VOC_CLASSES = [
    "background",
    "hold",
]
COLORMAP = [
    [0, 0, 0],
    [1, 1, 1],
]

class DataRoad(Dataset):
    def __init__(self, transform=None):
        super().__init__()

        self.transform = transform
        self.data_folder, self.label_folder = self.load_file_paths()
        self.normalize = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_file_paths(self):
        train_path = '/content/Data/'
        data_files = sorted(glob(os.path.join(train_path, 'images', '*.png')))
        label_files = sorted(glob(os.path.join(train_path, 'label', '*.png')))
        assert len(data_files) == len(label_files), "Number of data and label files do not match."
        return data_files, label_files

    def __len__(self):
        return len(self.data_folder)

    def _convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width), dtype=np.float32)
        for label_index, label in enumerate(COLORMAP):
            segmentation_mask[np.all(mask == label, axis=-1)] = label_index
        return segmentation_mask

    def image_loader(self, data_path, label_path):
        image = cv2.imread(data_path)
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path)
        label = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return data, label

    def __getitem__(self, idx):
        data_path, label_path = self.data_folder[idx], self.label_folder[idx]
        image, mask = self.image_loader(data_path, label_path)
        label = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        else:
            transformed_image, transformed_mask = image, label

        return transformed_image, torch.unsqueeze(transformed_mask, 0)  # Add channel dimension to mask
