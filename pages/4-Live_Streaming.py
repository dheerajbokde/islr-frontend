import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, VideoHTMLAttributes
import av
import threading
# import SRLDetector as srd
from scripts.SRLLiveDetector import *
from scripts.SLRClassifier import *

import numpy as np
import math
import random
import threading

frame_lock = threading.Lock()
record_lock = threading.Lock()

output = None


st.set_page_config(layout="wide")
# st.sidebar.title('Group 6 Demo')
st.title(':blue[Live Video Stream Processing]')

style_list = ['color', 'black and white']
predict_mode = st.sidebar.radio("Prediction Mode", ('Frame', 'Video'),index=0)
st.sidebar.markdown('---')

def process_data():
    global predictFlag, output
    predictFlag = st.session_state['15']
    if predictFlag:
        output = cv2.VideoWriter('./temp/output_video_from_file.mp4', -1, 30, (128, 128))
    else:
        if output is not None:
            output.release()
    # print(predictFlag , " Processing ....."+ str(len(RECORD)))
    # predictLiveStream(np.array(RECORD))

predictFlag = st.sidebar.checkbox("Predict Sign ...", value=False, key=15, on_change=process_data)
st.sidebar.markdown('---')

detection_confidence, tracking_confidence = 0.5, 0.5

def onChangeConfidence():
    global detection_confidence, tracking_confidence
    detection_confidence = st.session_state['1']
    tracking_confidence =  st.session_state['2']
    onConfidenceLevelChange(detection_confidence, tracking_confidence)

detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5, key=1, on_change=onChangeConfidence)
tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5, key=2, on_change=onChangeConfidence)
st.sidebar.markdown('---')



offset = 20
imgSize = 224
model_frame_size = 128

# Checkbox 
trackHand1, trackPose1, trackFace1 = False, False, False

# Tracking results
hand_results, pose_results, face_results = None, None, None
currentFrame = None
blackImage, handsTrackedblackImage = None, None
hands = None

# Prediction variables
MAX_FRAME_COUNT = 20
FRAME_SET = [] # of size MAX_FRAME_COUNT
RECORD = [] # number of frameset to be passed for the prediction
frame_counter = 0
record_counter = 0

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def callbackOriginal(frame):
    global hand_results, pose_results, face_results
    global currentFrame 
    global blackImage, handsTrackedblackImage
    global hands

    currentFrame = frame

    oh, ow = frame.height, frame.width

    img = frame.to_ndarray(format="bgr24")

    img, hand_results, pose_results, face_results, blackImage = getAnnotedOriginalImage(img, handsOn=trackHand1, poseOn=trackPose1, faceOn=trackFace1, black=True)
    hands, img, handsTrackedblackImage = findHandsOnOriginal(hand_results, img, tgh=224, tgw=224, flipType=True, trackHands=trackHand1, bboxVisible=True, black=True)
   
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def callbackTrasformed(frame):
    global currentFrame 
    global hand_results, pose_results, face_results
    global hands

    global FRAME_SET, MAX_FRAME_COUNT, RECORD, frame_counter, predicted_result, record_counter, output
    predicted_result = None

    # frame_counter = 0

    oh, ow = frame.height, frame.width

    # frame = currentFrame.to_ndarray(format="bgr24")
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
    # hands, frame, _ = srd.findHandsOnOriginal(results=hand_results, rawImg=frame, tgh=224, tgw=224, flipType=True, trackHands=trackHand2, bboxVisible=True)
    # imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)

    localblackImg = imgWhite
    global blackImage
    global handsTrackedblackImage

    if hands:
        if len(hands) > 1:
            # hand1 = hands[0]
            # hand2 = hands[1]
            # print("Two Hand Mode - ", hand1['bbox'], hand2['bbox'])
            # localblackImg = srd.getBlackImage(frame, poseOn=trackPose1, faceOn=trackFace1)
            localblackImg = blackImage
        elif len(hands) == 1:
            # Hand 1
            hand1 = hands[0]
            x1, y1, w1, h1 = hand1['bbox']
            
            imgCrop1 = handsTrackedblackImage[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]
            imgCropShape1 = imgCrop1.shape
            # print(imgCropShape1)
            imgResizeShape = None
            if imgCropShape1[0] > 0 and imgCropShape1[1] > 0:
                # print("One Hand Mode - ", imgCropShape1)
                aspectRatio1 = h1 / w1
                if aspectRatio1 > 1:
                    k1 = imgSize / h1
                    if math.ceil(k1 * w1) > imgSize:
                        wCal1 = math.floor(k1 * w1)
                    else:
                        wCal1 = math.ceil(k1 * w1)
                                     
                    imgResize1 = cv2.resize(imgCrop1, (wCal1, imgSize))
                    imgResizeShape = imgResize1.shape
                    wGap1 = math.ceil((imgSize - wCal1) / 2)
                    wmax = wCal1 + wGap1
                    if wmax > imgSize:
                        wmax = imgSize
                    # imgWhite[:, wGap1:wCal1 + wGap1] = imgResize1
                    imgWhite[:, wGap1:wmax] = imgResize1
                else:
                    k1 = imgSize / w1
                    if math.ceil(k1 * h1) > imgSize:
                        hCal1 = math.floor(k1 * h1)
                    else:
                        hCal1 = math.ceil(k1 * h1)

                    imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal1))
                    imgResizeShape = imgResize1.shape
                    hGap = math.ceil((imgSize - hCal1) / 2)
                    hmax = hCal1 + hGap
                    if hmax > imgSize:
                        hmax = imgSize
                    # imgWhite[hGap:hCal1 + hGap, :] = imgResize1
                    imgWhite[hGap:hmax, :] = imgResize1
                localblackImg = imgWhite
            # blackImg, _, _, _ = srd.getAnnotedOriginalImage(blackImg, handsOn=True, poseOn=False, faceOn=False)
            # blackImg = srd.plotOnBlackImage(hand_results=hand_results, pose_results=pose_results, face_results=face_results, rawImg=imgWhite, poseOn=False, faceOn=False)
        else:
            localblackImg = imgWhite
    else:
        localblackImg = imgWhite

    if predictFlag:
        if predict_mode == 'Frame':
            print('Image Mode....')
            # predictLable(localblackImg)
        elif predict_mode == 'Video':
            resized_frame = image_resize(localblackImg, width=model_frame_size, height=model_frame_size, inter=cv2.INTER_AREA)
            if output is not None:
                output.write(resized_frame)
    else:
        if output is not None:
            output.release()
    
 
    if predicted_result is not None:
        print(predicted_result)

    return av.VideoFrame.from_ndarray(localblackImg, format="bgr24").reformat(width=ow, height=oh, format="bgr24")
    # return av.VideoFrame.from_ndarray(blackImage, format="bgr24").reformat(width=ow, height=oh, format="bgr24")



col1, col2 = st.columns(2)
with col1:
    st.header("Live Stream...")
    ctx1 = webrtc_streamer(key="Original", video_frame_callback=callbackOriginal, 
                           video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, style={"width": "100%"}, muted=True))
    
    trackHand1 = st.sidebar.checkbox("Track Hands", value=True,key=11)
    trackPose1 = st.sidebar.checkbox("Track Pose", value=False, key=12)
    trackFace1 = st.sidebar.checkbox("Track Face", value=False, key=13)

    col1.markdown('---')

with col2:
    st.header("Transformed Stream...")
    ctx2 = webrtc_streamer(key="Transformed", video_frame_callback=callbackTrasformed, 
                           video_html_attrs=VideoHTMLAttributes(
                                autoPlay=True, controls=True, style={"width": "100%"}, muted=True), async_transform = True)

    col2.markdown('---')
