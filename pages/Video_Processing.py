import streamlit as st
import cv2

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
st.title(':blue[Video Processing]')

col1, col2 = st.columns(2)
video = None

# Uploading the File to the Page
with col1:
    st.header("Upload Video")
    uploaded_video = st.file_uploader("Choose video", type=["mp4"])
    st.markdown("""---""")

    st.header("Capture Video")
    captured_video = st.camera_input("First, record a video...")


def process_video(video):
    col2.write("VIDEO SIGN TEXT")


with col2:
    st.header("Video Preview")
    if uploaded_video is not None:
        video = uploaded_video
    elif captured_video is not None:
        video = captured_video

    if video is not None: 
        st.write("Video Uploaded Successfully")
        vid = video.name
        with open(vid, mode='wb') as f:
            # save video to disk
            f.write(video.read()) 
            # display file name
            #st.markdown(f"""### Files- {vid}""",unsafe_allow_html=True)
            # load video from disk
            #vidcap = cv2.VideoCapture(vid) 
            st.video(video)
            st.button('Predict Sign Language',on_click=process_video,args=(video,))
    else:
        st.write("Make sure you load video in MP4 format.")
