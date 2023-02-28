import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image

from src.dataset import SegmentationDataset
from src.utils import mask_visualization

MASK_BG = 1
train_tfm = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomScale(),
        A.Rotate(border_mode=cv2.BORDER_CONSTANT, mask_value=MASK_BG),
        A.RandomBrightnessContrast(p=0.2),
        A.SmallestMaxSize(224),
        A.RandomCrop(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

dataset = SegmentationDataset(
    image_dir=r"data\images",
    mask_dir=r"data\annotations\trimaps",
    transform=train_tfm,
)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def imagenet_denorm(x):
    """x: array-like with shape (..., H, W, C)"""
    return x * imagenet_std + imagenet_mean


for _ in range(3):
    img, mask = dataset[0]
    print(mask.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(imagenet_denorm(img.numpy().transpose(1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(mask.numpy())
    plt.xticks([])
    plt.yticks([])
    plt.show()
