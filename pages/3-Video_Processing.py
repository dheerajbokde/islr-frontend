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
    video = st.file_uploader("Choose video", type=["mp4"])
    if video is not None:
        with open(video.name, mode='wb') as f:
            # save video to disk
            f.write(video.read()) 
    st.markdown("""---""")

    st.header("Record Video")
    run = st.checkbox('Start recording a video...')
    while run:
        cam_video = cv2.VideoCapture(0)
        frame_size = (640, 480)
        video_file = 'testvideo.mp4'
        #fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = -1
        writer = cv2.VideoWriter(video_file, fourcc, 20, frame_size)
        window_name = 'Video'
        while (cam_video.isOpened()):
            ret, frame = cam_video.read()
            if ret:
                writer.write(frame)
                cv2.imshow(window_name, frame)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(1) & 0xFF == 27:
                    run,ret = False,False
                    break
            else: 
                break
        cam_video.release()
        writer.release()
        cv2.destroyAllWindows()
        #cv2.destroyWindow(window_name)
        vid = open(video_file, 'rb')
        video = vid.read()


def process_video(video):
    col2.write("VIDEO SIGN TEXT")


with col2:
    st.header("Video Preview")
    if video is not None: 
        st.write("Video Uploaded Successfully")
        st.video(video)
        st.button('Predict Sign Language',on_click=process_video,args=(video,))
    else:
        st.write("Make sure you load video in MP4 format.")
