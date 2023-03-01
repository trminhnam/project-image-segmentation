# Reference: https://albumentations.ai/docs/examples/showcase/, https://albumentations.ai/docs/examples/example_kaggle_salt/
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(mask_bg=1):
    # MASK_BG = 2 - 1
    train_tfm = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(),
            A.Rotate(border_mode=cv2.BORDER_CONSTANT, mask_value=mask_bg),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1
            ),
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
            A.CLAHE(p=0.2),
            # A.RandomShadow(p=0.2),
            # A.RandomSnow(p=0.2),
            # A.RandomRain(p=0.2),
            # A.RandomFog(p=0.2),
            # A.RandomSunFlare(p=0.2),
            # A.RandomSunFlare(p=0.2),
            A.SmallestMaxSize(224),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_tfm


def get_val_transforms():
    val_tfm = A.Compose(
        [
            A.SmallestMaxSize(224),
            A.CenterCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return val_tfm
