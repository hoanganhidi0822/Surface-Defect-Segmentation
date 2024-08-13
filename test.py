import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.ResUnet_CBAM import ResUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
checkpoint_path = "/content/drive/MyDrive/Unet/best_model.pth"
model = ResUnet(1)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load and preprocess the image
image_path = '/content/drive/MyDrive/Unet/image_703.png'
image_ = cv2.imread(image_path)
image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)

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

normalize = A.Compose([
    A.Resize(width=640, height=480),
    EnhanceBlackWhitePoints(p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

transformed = normalize(image=image)
image = transformed['image'].unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    x = image.to(device).float()
    y_hat = model(x)
    y_hat = torch.sigmoid(y_hat)

# Apply threshold to get binary segmentation
threshold = 0.5
segmentation_map = (y_hat.squeeze().cpu().numpy() > threshold).astype(np.uint8)

# Define class labels and colormap
VOC_COLORMAP = [
    [0, 0, 0],      # background (black)
    [200, 0, 240],  # hold (purple)
]

# Display the results with colormap
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_.astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

colored_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
for i, color in enumerate(VOC_COLORMAP):
    colored_map[segmentation_map == i] = np.array(color)

plt.subplot(1, 3, 2)
plt.imshow(colored_map)
plt.title('Predicted Segmentation Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_.astype(np.uint8))
plt.imshow(colored_map, alpha=0.5)
plt.title('Overlay')
plt.axis('off')

overlay_path = '/content/drive/MyDrive/Unet/saved_image.png'
plt.savefig(overlay_path, bbox_inches='tight')
plt.show()

# Use the segmentation map for further processing
def visualize_class_value(segmentation_map, class_value=1):
    binary_mask = np.where(segmentation_map == class_value, 255, 0).astype(np.uint8)
    return binary_mask

def save_visualization(binary_mask, output_path):
    binary_image = Image.fromarray(binary_mask)
    binary_image.save(output_path)
    print(f"Visualization saved to {output_path}")

def display_visualization(binary_mask):
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Class Value Visualization')
    plt.axis('off')
    plt.show()

def draw_bounding_boxes_and_annotate(binary_mask, original_image, pixels_per_cm=18):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        longer_edge_pixels = max(w, h)
        longer_edge_cm = longer_edge_pixels / pixels_per_cm
        color = (0, 0, 255) if longer_edge_cm > 1.1 else (0, 255, 0)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 1)
    
    return original_image

# Create a binary mask for the class value
binary_mask = visualize_class_value(segmentation_map, class_value=1)

# Save the binary mask as an image
binary_mask_path = '/content/drive/MyDrive/Unet/binary_mask.png'
save_visualization(binary_mask, binary_mask_path)

# Read the binary mask as a grayscale image for processing
binary_mask_for_bboxes = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

# Draw bounding boxes and annotate with the longer edge length in cm
bbox_image = draw_bounding_boxes_and_annotate(binary_mask_for_bboxes, image_, pixels_per_cm=23.5)

# Display the image with bounding boxes and annotations
plt.imshow(bbox_image)
plt.title('Bounding Boxes with Longer Edge Annotation in cm')
plt.axis('off')
plt.show()

bbox_output_path = '/content/drive/MyDrive/Unet/image_with_bboxes.png'
cv2.imwrite(bbox_output_path, bbox_image)
print(f"Bounding boxes visualization saved to {bbox_output_path}")
