import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

import wandb
from src.augmentation import get_train_transforms, get_val_transforms
from src.dataset import SegmentationDataset
from src.model import UNet
from src.utils import (
    evaluate_fn,
    get_data_loader,
    load_checkpoint,
    plot_metrics,
    save_checkpoint,
    train_fn,
    wandb_init,
    wandb_log,
    wandb_save,
)

config = {}
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    # init wandb
    if config.get("wandb", False) and config.get("wandb_project", None):
        wandb_init(project_name=config["wandb_project"], config=config)

    # init save path
    save_dir = config.get("save_dir", "output")
    os.makedirs(save_dir, exist_ok=True)

    # load transforms
    train_tfm = get_train_transforms(mask_bg=config.get("mask_bg", 1))
    val_tfm = get_val_transforms()

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3, out_channels=3, activation=config.get("activation", "relu")
    )
    if config.get("model_load_path", None):
        load_checkpoint(config["model_load_path"], model)
    model.to(device)

    # load optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # load dataset
    train_loader, test_loader = get_data_loader(config)

    # train
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_dice_scores = []
    for epoch in range(config["epochs"]):
        print(f"Epoch: {epoch+1}/{config['epochs']}")
        train_loss = train_fn(train_loader, model, optimizer, criterion, scaler, device)
        test_loss, accuracy, dice_score = evaluate_fn(
            test_loader, model, criterion, device
        )
        print(
            f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {accuracy:.4f}, Test dice_score: {dice_score:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(test_loss)
        val_accuracies.append(accuracy)
        val_dice_scores.append(dice_score)
        wandb_log(
            {
                "train_loss": train_loss,
                "val_loss": test_loss,
                "val_accuracy": accuracy,
                "val_dice_score": dice_score,
            },
            epoch,
        )

        if (
            config.get("save_every", -1) != -1
            and (epoch + 1) % config["save_every"] == 0
        ):
            save_checkpoint(
                model.state_dict(),
                checkpoint_path=os.path.join(
                    save_dir,
                    config.get("model_save_name", "model.pth").split(".")[0]
                    + f"_{epoch+1}.pth",
                ),
            )
            wandb_save(save_dir)

        print()

    # save last model
    save_checkpoint(
        model.state_dict(),
        checkpoint_path=os.path.join(
            save_dir,
            config.get("model_save_name", "model.pth").split(".")[0] + "_final.pth",
        ),
    )

    plot_metrics(
        [train_losses, val_losses],
        ["train", "val"],
        "Epochs",
        "Loss",
        "Train and Val Losses",
        os.path.join(save_dir, "losses.png"),
    )

    plot_metrics(
        [val_accuracies],
        ["val"],
        "Epochs",
        "Accuracy",
        "Val Accuracy",
        os.path.join(save_dir, "accuracy.png"),
    )

    plot_metrics(
        [val_dice_scores],
        ["val"],
        "Epochs",
        "Dice Score",
        "Val Dice Scores",
        os.path.join(save_dir, "dice_score.png"),
    )
    wandb_save(save_dir)
