import torch
import os
import torchmetrics
import torch.nn as nn
from model.ResUnet_CBAM import ResUnet
from dataset import dataloader
from ultils import AverageMeter, accuracy_function
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt

mean_iou = 0
mean_loss = float('inf')
patience = 5
patience_counter = 0

# Define transforms
class EnhanceBlackWhitePoints(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(EnhanceBlackWhitePoints, self).__init__(always_apply, p)

    def apply(self, image, **params):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        low_value = np.percentile(l, 2)
        high_value = np.percentile(l, 98)
        l_stretched = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        l_stretched = cv2.convertScaleAbs(l_stretched, alpha=255/(high_value-low_value), beta=-low_value*255/(high_value-low_value))
        enhanced_lab = cv2.merge((l_stretched, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_image

train_transform = A.Compose([
    A.Resize(width=640, height=480),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    EnhanceBlackWhitePoints(p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(width=640, height=480),
    EnhanceBlackWhitePoints(p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Load data
batch_size = 12
n_workers = 4
print("num_workers =", n_workers)

dataset = dataloader.DataRoad(train_transform)
print(len(dataset))

total_samples = len(dataset)
train_split = int(0.8 * total_samples)
train_set, val_set = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split])

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
testloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# Model
model = ResUnet(1).to(device)  # Assuming binary segmentation

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00009)
n_eps = 18

# Metrics
dice_fn = torchmetrics.Dice(num_classes=2, average="macro").to(device)
iou_fn = torchmetrics.JaccardIndex(num_classes=2, task="binary", average="macro").to(device)

# Meters
acc_meter = AverageMeter()
train_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training and Validation
for ep in range(1, 1 + n_eps):
    acc_meter.reset()
    train_loss_meter.reset()
    dice_meter.reset()
    iou_meter.reset()

    # Training
    model.train()
    for batch_id, (x, y) in enumerate(tqdm(trainloader), start=1):
        optimizer.zero_grad()
        n = x.shape[0]
        x = x.to(device).float()
        y = y.to(device).float()  # BCEWithLogitsLoss expects float targets
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat_mask = torch.sigmoid(y_hat) > 0.5  # Apply sigmoid and threshold for binary mask
            dice_score = dice_fn(y_hat_mask, y.long())
            iou_score = iou_fn(y_hat_mask, y.long())
            accuracy = accuracy_function(y_hat_mask, y.long())

            train_loss_meter.update(loss.item(), n)
            iou_meter.update(iou_score.item(), n)
            dice_meter.update(dice_score.item(), n)
            acc_meter.update(accuracy.item(), n)

    train_losses.append(train_loss_meter.avg)
    train_accuracies.append(acc_meter.avg)

    print("Epoch {}, train loss = {}, accuracy = {}, IoU = {}, dice = {}".format(
        ep, train_loss_meter.avg, acc_meter.avg, iou_meter.avg, dice_meter.avg
    ))

    # Validation
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    val_dice_meter = AverageMeter()
    val_iou_meter = AverageMeter()

    with torch.no_grad():
        for x, y in tqdm(testloader):
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            loss = criterion(y_hat, y)

            y_hat_mask = torch.sigmoid(y_hat) > 0.5
            dice_score = dice_fn(y_hat_mask, y.long())
            iou_score = iou_fn(y_hat_mask, y.long())
            accuracy = accuracy_function(y_hat_mask, y.long())

            val_loss_meter.update(loss.item(), x.size(0))
            val_iou_meter.update(iou_score.item(), x.size(0))
            val_dice_meter.update(dice_score.item(), x.size(0))
            val_acc_meter.update(accuracy.item(), x.size(0))

    val_losses.append(val_loss_meter.avg)
    val_accuracies.append(val_acc_meter.avg)

    print("Validation - Epoch {}, val loss = {}, accuracy = {}, IoU = {}, dice = {}".format(
        ep, val_loss_meter.avg, val_acc_meter.avg, val_iou_meter.avg, val_dice_meter.avg
    ))

    # Early Stopping
    if val_iou_meter.avg > mean_iou and val_loss_meter.avg < mean_loss:
        torch.save(model.state_dict(), "/content/drive/MyDrive/Unet/best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

    if ep == n_eps:
        torch.save(model.state_dict(), "/content/drive/MyDrive/Unet/last_model.pth")

    mean_iou = val_iou_meter.avg
    mean_loss = val_loss_meter.avg

# Plotting loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.savefig('/content/drive/MyDrive/Unet/training_metrics.png')
plt.show()
