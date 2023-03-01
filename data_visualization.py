import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image

from src.dataset import SegmentationDataset
from src.utils import mask_visualization
from src.augmentation import get_train_transforms, get_val_transforms

MASK_BG = 2 - 1
train_tfm = get_train_transforms(mask_bg=MASK_BG)

dataset = SegmentationDataset(
    image_dir=r"data\images",
    mask_dir=r"data\annotations\trimaps",
    transform=get_val_transforms(),
)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def imagenet_denorm(x):
    """x: array-like with shape (..., H, W, C)"""
    return x * imagenet_std + imagenet_mean

# plot 4 images with their masks side by side
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(4):
    idx = np.random.randint(0, len(dataset))
    img, mask = dataset[idx]
    ax[0, i].imshow(imagenet_denorm(img.numpy().transpose(1, 2, 0)))
    ax[0, i].set_title("Image")
    ax[0, i].set_xticks([])
    ax[0, i].set_yticks([])

    ax[1, i].imshow(mask.numpy())
    ax[1, i].set_title("Mask")
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])
plt.tight_layout()
plt.savefig("sample_images/data_visualization.png")
plt.show()

# for i in range(3):
#     idx = np.random.randint(0, len(dataset))
#     img, mask = dataset[idx]
#     print(mask.shape)
#     plt.subplot(1, 2, 1)
#     plt.imshow(imagenet_denorm(img.numpy().transpose(1, 2, 0)))
#     plt.title("Image")
#     plt.xticks([])
#     plt.yticks([])
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask.numpy())
#     plt.title("Mask")
#     plt.xticks([])
#     plt.yticks([])
    
#     plt.tight_layout()
#     plt.savefig(f"sample_images/data_visualization_{i}.png")
#     plt.show()
