import numpy as np
from PIL import Image, ImageOps


def open_image(image_buffer):
    """
        image_buffer : The image buffer from streamlit st.upload_file API

    Returns:
       A 2D numpy array containing the grayscaled image
    """
    rgba_image = Image.open(image_buffer)
    # print("rgba_image.shape:",rgba_image.size)
    grayscale_image = ImageOps.grayscale(rgba_image)
    print(grayscale_image.size)
    grayscale_image_array = np.array(grayscale_image)
    # print("grayscale_image_array.shape:",grayscale_image_array.shape)
    return grayscale_image_array