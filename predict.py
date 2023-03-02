import os

import numpy as np
import torch
import yaml
from PIL import Image

from src.model import UNet
from src.utils import load_checkpoint


def predict(model, image, device):
    model.eval()
    image = image / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    output = torch.argmax(output, dim=1)
    output = output.squeeze(0).cpu().numpy()
    return output


def visualize(image, mask, num_classes=3, background_idx=1, alpha=0.2):
    # for each class in the mask, draw the corresponding color
    # on the image
    colors = [np.random.randint(0, 255, 3) for _ in range(num_classes)]

    for i in range(num_classes):
        if i == background_idx:
            continue
        image[mask == i] = image[mask == i] * (1 - alpha) + np.array(colors[i]) * alpha

    # display the image
    image2display = Image.fromarray(image.astype("uint8"))
    image2display.show()

    return image


if __name__ == "__main__":
    config = {}
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=config.get("num_classes", 3))
    load_checkpoint(config["model_load_path"], model)
    model = model.to(device)

    image_path = "./sample_images/cat.jpg"
    image = np.array(Image.open(image_path))
    mask = predict(model, image, device)
    result = visualize(image, mask, num_classes=config.get("num_classes", 3), alpha=0.5)

    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Image.fromarray(result.astype("uint8")).save(
        os.path.join(save_dir, os.path.basename(image_path))
    )
