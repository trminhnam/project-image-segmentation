import streamlit as st
import requests
from io import BytesIO
import numpy as np
from PIL import Image

# import mrcnn.model as modellib
# from mrcnn import visualize

# # Define configuration for Mask R-CNN model
# class SegmentationConfig():
#     NAME = "coco"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + background

# config = SegmentationConfig()
# model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./logs")

# # Load pre-trained weights from COCO dataset
# model.load_weights("./mask_rcnn_coco.h5", by_name=True)

# # Define function to perform segmentation on input image
# def segment_image(image):
#     image_array = np.array(image)
#     results = model.detect([image_array])
#     r = results[0]
#     mask = r['masks'][:,:,0]
#     return mask

def segment_image(image):
    return image

# Define Streamlit app
st.title('Image Segmentation App')

# Ask user for input image or link
input_type = st.radio('Input type', ('Image file', 'Image URL'))

if input_type == 'Image file':
    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Input image')
        except Exception as e:
            st.error('Error: {}'.format(e))
            st.stop()
        
        mask = segment_image(image)
        st.image(mask, caption='Segmentation map')
        

else:
    image_url = st.text_input('Enter the URL of an image')
    if st.button('Segment'):
        if image_url != '':
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Input image')
            
            mask = segment_image(image)
            st.image(mask, caption='Segmentation map')
        else:
            st.warning('Please enter an image URL to continue')
            st.stop()
