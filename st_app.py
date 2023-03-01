import torch
import streamlit as st
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from predict import predict, visualize
from src.model import UNet
from src.utils import load_checkpoint
import yaml

config = {}
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


@st.cache(allow_output_mutation=True)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=config.get("num_classes", 3))
    load_checkpoint(config["model_load_path"], model)
    model = model.to(device)
    return model, device


def segment_image(image):
    global model, device
    image = np.array(image)
    mask = predict(model, image, device)
    mask = visualize(image, mask, num_classes=config.get("num_classes", 3), alpha=0.5)
    return mask


# Define Streamlit app
st.title("Image Segmentation App")

with st.spinner("Loading model into memory..."):
    model, device = load_model()

# Ask user for input image or link
input_type = st.radio("Input type", ("Image file", "Image URL"))

if input_type == "Image file":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input image")
        except Exception as e:
            st.error("Error: {}".format(e))
            st.stop()

        mask = segment_image(image)
        st.image(mask, caption="Segmentation map")


else:
    image_url = st.text_input("Enter the URL of an image")
    if st.button("Segment"):
        if image_url != "":
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Input image")

            mask = segment_image(image)
            st.image(mask, caption="Segmentation map")
        else:
            st.warning("Please enter an image URL to continue")
            st.stop()
