from PIL import Image
import numpy as np 
import streamlit as st 
from keras.models import load_model
import keras.utils as image
import os

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
st.title(':blue[Image Processing]')

col1, col2 = st.columns(2)
picture = None
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']

# Function to load an images
def load_image(image):
    img = Image.open(image)
    img_array = np.array(img)
    return img_array, img


def process_image(image_name):
    #print('Process image: ', image_name)
    model = load_model('artifacts/model/cnn_isl_img_model.h5')
    img = image.load_img(image_name, target_size = (28,28))
    # convert image into array for prediction
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis = 0)
    # predict image using model
    result = model.predict(test_image).argmax()
    #col2.write(labels[result])
    predicted_text = labels[result]
    pred_html_str = f"""
        <style>
        p.a {{
            font: bold 45px Courier;
        }}
        </style>
        <p class="a">{predicted_text}</p>
    """
    col2.markdown(pred_html_str, unsafe_allow_html=True)

with col1:
    st.header("Upload Image")
    uploaded_image = st.file_uploader(label="Upload JPG/PNG Image", type=['jpg', 'png'])
    st.markdown("""---""")
    st.header("Take Picture")
    captured_image = st.camera_input("First, take a picture...")


with col2:
    st.header("Image Preview")
    if uploaded_image is not None:
        picture = uploaded_image
    elif captured_image is not None:
        picture = captured_image

    if picture is not None:
        img_array, img = load_image(picture)
        # save picture to disk
        image_name = 'test.jpg'
        img.save(image_name)
        #print(img_array)
        st.write("Image Uploaded Successfully")
        st.image(img, use_column_width=True)
        st.button('Predict Sign Language',on_click=process_image,args=(image_name,))
    else:
        st.write("Make sure you load image in JPG/PNG format.")
