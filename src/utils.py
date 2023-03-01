import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm.auto import tqdm


def mask_visualization(img, mask, alpha=0.5):
    """
    Visualize mask on top of image
    Number of channels of mask is the number of instances
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def train_test_split_dataset(dataset, val_split=0.25):
    """Split PyTorch dataset into train and test set.

    Reference: https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5

    Args:
        `dataset` (torch.utils.data.Dataset): dataset to split
        `val_split` (float, optional): percentage of dataset to use for validation. Defaults to 0.25.

    Returns:
        `dict`: dictionary with keys 'train' and 'test' containing the train and test datasets
    """
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["test"] = Subset(dataset, test_idx)
    return datasets


def get_increment_path(path, increment=1):
    """Get path with incremented number.

    Args:
        `path` (str): path to increment
        `increment` (int, optional): number to increment by. Defaults to 1.

    Returns:
        `str`: incremented path
    """
    counter = 0
    new_path = path
    while os.path.exists(new_path):
        counter += increment
        new_path = path.split(".")[0] + f"_{counter}" + "." + path.split(".")[1]
    return new_path


def save_masks_and_preds_to_img(masks, preds, folder="saved_images/"):
    """Save batch of images, masks, and predictions to disk.

    Args:
        `masks` (torch.Tensor): batch of ground truth masks in shape (N, H, W)
        `preds` (torch.Tensor): batch of model predictions in shape (N, C, H, W)
        `folder` (str, optional): folder to save images to. Defaults to "saved_images/"
    """
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    preds = np.argmax(preds, axis=1)

    # scale mask and pred to 0-255
    max_value = np.max(masks)
    masks = (masks / max_value * 255.0).astype(np.uint8)
    preds = (preds / max_value * 255.0).astype(np.uint8)

    # concatenate mask and pred
    mask_pred = np.concatenate([masks, preds], axis=2)

    # concat all images in mask_pred to a long image along batch dimension
    mask_pred = np.concatenate(mask_pred, axis=0)

    # save image
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, "mask_pred.png")
    save_path = get_increment_path(save_path)
    cv2.imwrite(save_path, mask_pred)


def save_checkpoint(state, checkpoint_path="my_checkpoint.pth.tar"):
    """Save model checkpoint.

    Args:
        `state` (dict): model state dictionary
        `checkpoint_path` (str, optional): path to save checkpoint. Defaults to "my_checkpoint.pth.tar".
    """
    print(f"=> Saving checkpoint at {checkpoint_path}")
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, model):
    """Load model checkpoint.

    Args:
        `checkpoint_path` (str): path to checkpoint
        `model` (torch.nn.Module): model to load checkpoint into

    Returns:
        `dict`: model state dictionary
    """
    print(f"=> Loading checkpoint at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return checkpoint


def calculate_accuracy(outputs, labels):
    """Calculate accuracy of model predictions.

    Args:
        `outputs` (torch.Tensor): model predictions (logits) in shape (N, C, H, W)
        `labels` (torch.Tensor): ground truth labels in shape (N, H, W)

    Returns:
        `float`: accuracy
    """
    outputs = outputs.argmax(dim=1)
    return torch.sum(outputs == labels).item() / torch.numel(outputs)


def calculate_dice_score(outputs, labels, num_classes=3):
    """Calculate dice score of model predictions.

    Args:
        `outputs` (torch.Tensor): model predictions (logits) in shape (N, C, H, W)
        `labels` (torch.Tensor): ground truth labels in shape (N, H, W)

    Returns:
        `float`: dice score
    """
    outputs = outputs.argmax(dim=1).cpu()
    targets = labels.cpu()
    dice_score = 0
    for cls in range(num_classes):
        i = outputs == cls
        t = targets == cls
        intersection = (i & t).sum()
        union = (i | t).sum()
        if not union:
            dice_score += 1 if not intersection else 0
        else:
            dice_score += 2.0 * intersection / union
    return dice_score / num_classes


def train_fn(train_loader, model, optimizer, criterion, scaler, device):
    pbar = tqdm(train_loader)
    losses = []

    for batch_idx, (image, target) in enumerate(pbar):
        image = image.to(device)
        target = target.to(device).long()

        with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output.view(-1, 3), target.view(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_description(f"Train | loss: {loss.item():.4f}")
        losses.append(loss.item())

    return np.mean(losses)


def evaluate_fn(val_loader, model, criterion, device):
    model = model.eval()
    model = model.to(device)

    pbar = tqdm(val_loader)
    losses = []
    accuracies = []
    dice_scores = []

    visualization_idx = np.random.randint(0, len(val_loader))

    for batch_idx, (image, target) in enumerate(pbar):
        # move to device
        image = image.to(device)
        target = target.to(device).long()

        # calculate loss
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output.view(-1, 3), target.view(-1))

        # visualize some predictions
        if batch_idx == visualization_idx:
            save_masks_and_preds_to_img(target, output)

        # calculate accuracy and dice score
        accuracy = calculate_accuracy(output, target)
        accuracies.append(accuracy)

        dice_score = calculate_dice_score(output, target)
        dice_scores.append(dice_score)

        # update progress bar
        pbar.set_description(
            f"Val | loss: {loss.item():.4f}, accuracy: {accuracy:.4f}, dice_score: {dice_score:.4f}"
        )
        losses.append(loss.detach().cpu().numpy())

    loss = np.mean(losses)
    acc = np.mean(accuracies)
    dice_score = np.mean(dice_scores)
    return loss, acc, dice_score
