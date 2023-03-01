import os
import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

from src.dataset import SegmentationDataset
from src.model import UNet
from src.utils import (
    evaluate_fn,
    load_checkpoint,
    save_checkpoint,
    train_fn,
    train_test_split_dataset,
)

config = {}
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)

if __name__ == "__main__":
    MASK_BG = 2 - 1
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
    val_tfm = A.Compose(
        [
            A.SmallestMaxSize(224),
            A.CenterCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3)
    if config["load_model"]:
        load_checkpoint(config["model_load_path"], model)
    model.to(device)

    # load optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # load dataset
    dataset = SegmentationDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        transform=train_tfm,
    )

    train_test_dataset = train_test_split_dataset(dataset, val_split=0.25)
    train_dataset = train_test_dataset["train"]
    test_dataset = train_test_dataset["test"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
    )

    # train
    for epoch in range(config["epochs"]):
        print(f"Epoch: {epoch+1}/{config['epochs']}")
        train_loss = train_fn(train_loader, model, optimizer, criterion, scaler, device)
        test_loss, accuracy, dice_score = evaluate_fn(
            test_loader, model, criterion, device
        )
        print(
            f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {accuracy:.4f}, Test dice_score: {dice_score:.4f}"
        )

        if (
            config.get("save_every", -1) != -1
            and (epoch + 1) % config["save_every"] == 0
        ):
            save_checkpoint(
                model.state_dict(),
                checkpoint_path=config["model_save_path"].split(".")[0]
                + f"_{epoch+1}.pth",
            )

        print()
