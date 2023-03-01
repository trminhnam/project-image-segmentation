import os
import yaml
import torch
import numpy as np
from PIL import Image

from src.model import UNet
from src.utils import load_checkpoint


config = {}
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=config.get("num_classes", 3))
load_checkpoint(config["model_load_path"], model)
model = model.to(device)

def predict(image):
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

def visualize(image, mask, num_classes=3):
    # for each class in the mask, draw the corresponding color
    # on the image
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(num_classes):
        image[mask == i] = colors[i]
    image = Image.fromarray(image.astype("uint8"))
    image.show()
    

image = np.array(Image.open("./sample_images/cat.jpg"))
mask = predict(image)
visualize(image, mask)