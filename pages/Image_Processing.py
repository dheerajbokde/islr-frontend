from PIL import Image
import numpy as np 
import streamlit as st 


# Function to Read and Manupilate Images
def load_image(image):
    img = Image.open(image)
    img_array = np.array(img)
    return img_array, img

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
st.title(':blue[Image Processing]')

col1, col2 = st.columns(2)
picture = None

def process_image(img):
    col2.write("IMAGE SIGN TEXT")


# Uploading the File to the Page
col1.header("Upload Image")
uploaded_image = col1.file_uploader(label="Upload JPG/PNG Image", type=['jpg', 'png'])

col1.markdown("""---""")

col1.header("Take Picture")
captured_image = col1.camera_input("First, take a picture...")

col2.header("Image for processing")
if uploaded_image is not None:
    picture = uploaded_image
elif captured_image is not None:
    picture = captured_image

if picture is not None: 
    img_array, img = load_image(picture)
    col2.write("Image Uploaded Successfully")
    col2.image(img, use_column_width=True)
    col2.button('Predict Sign Language',on_click=process_image,args=(img,))
else:
    col2.write("Make sure you image is in JPG/PNG Format.")

# Uploading the File to the Page
#uploadFile = st.file_uploader(label="Upload Image", type=['jpg', 'png'])

# Checking the Format of the page
# if uploadFile is not None:
#     img, im = load_image(uploadFile)
#     # st.image(img)
#     st.write("Image Uploaded Successfully")
#     col2.image(im, use_column_width=True)

    # col1, col2 = st.columns(2)

    # col1.header("Original")
    # col1.file_uploader(label="Upload JPG Image", type=['jpg', 'png'])
    # #col1.image(img, use_column_width=True)

    # grayscale = im.convert('LA')
    # picture = col2.camera_input("First, take a picture...")
    # col2.header("Grayscale")
    # col2.image(grayscale, use_column_width=True)
# else:
#     st.write("Make sure you image is in JPG/PNG Format.")