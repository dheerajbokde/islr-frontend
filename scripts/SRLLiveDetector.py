import cv2
import mediapipe as mp
import numpy as np
import pandas as pd



# Drawing specs
mpDraw = mp.solutions.drawing_utils
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=0, color=[255, 0, 0])
connection_drawing_spec = mpDraw.DrawingSpec(thickness=1, color=[4, 244, 4])
mp_drawing_styles = mp.solutions.drawing_styles

# Hands Landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Pose Landmarks
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
points = mpPose.PoseLandmark

# Face Mesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, 
                               max_num_faces=1, 
                               min_detection_confidence=0.5, 
                               min_tracking_confidence=0.5)

b_offset = 20
type_offset = 30

def onConfidenceLevelChange(min_detection_confidence, min_tracking_confidence):
    # print(min_detection_confidence, min_tracking_confidence)
    global hands, pose, faceMesh
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)
    
    pose = mpPose.Pose(static_image_mode=False,
                    smooth_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)
    
    faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, 
                               max_num_faces=1, 
                               min_detection_confidence=min_detection_confidence, 
                               min_tracking_confidence=min_tracking_confidence)

def generate_empty_image(h=224, w=224, c=3):
    return np.zeros(shape=(h,w,c), dtype=np.uint8) * 255

def getAnnotedOriginalImage(rawImg, tgh=224, tgw=224, handsOn=True, poseOn=False, faceOn=False, black=False):
    imgRGB = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(imgRGB)

    pose_results, face_results = None, None
    if poseOn:
        pose_results = pose.process(imgRGB)
    if faceOn:
        face_results = faceMesh.process(imgRGB)

    blackImg = generate_empty_image(tgh, tgw)
    
    if handsOn:
        if hand_results.multi_hand_landmarks:
            for hid, handLms in enumerate(hand_results.multi_hand_landmarks):
                mpDraw.draw_landmarks(rawImg, handLms, mpHands.HAND_CONNECTIONS)
                if black:
                    mpDraw.draw_landmarks(blackImg, handLms, mpHands.HAND_CONNECTIONS, 
                                            landmark_drawing_spec=drawing_spec, 
                                            connection_drawing_spec=connection_drawing_spec)

    if poseOn and pose_results.pose_landmarks:
        mpDraw.draw_landmarks(rawImg, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        if black:
                mpDraw.draw_landmarks(blackImg, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec, 
                                                                connection_drawing_spec=connection_drawing_spec)

    if faceOn and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mpDraw.draw_landmarks(rawImg, landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(rawImg, landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            if black:
                mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    return rawImg, hand_results, pose_results, face_results, blackImg

def getBlackImage(rawImg, tgh=224, tgw=224, poseOn=False, faceOn=False):

    imgRGB = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(imgRGB)
    pose_results = pose.process(imgRGB)
    face_results = faceMesh.process(imgRGB)

    blackImg = generate_empty_image(tgh, tgw)

    if hand_results.multi_hand_landmarks:
        for hid, handLms in enumerate(hand_results.multi_hand_landmarks):
            mpDraw.draw_landmarks(blackImg, handLms, mpHands.HAND_CONNECTIONS, 
                                  landmark_drawing_spec=drawing_spec, 
                                  connection_drawing_spec=connection_drawing_spec)

    if poseOn and pose_results.pose_landmarks:
        mpDraw.draw_landmarks(blackImg, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS, 
                              landmark_drawing_spec=drawing_spec, 
                              connection_drawing_spec=connection_drawing_spec)

    if faceOn and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, 
                                  connections=mpFaceMesh.FACEMESH_CONTOURS, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, 
                                  connections=mpFaceMesh.FACEMESH_TESSELATION, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    return blackImg

def plotOnBlackImage(hand_results, pose_results, face_results, rawImg, tgh=224, tgw=224,  handsOn=True, poseOn=False, faceOn=False):

    # imgRGB = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
    # hand_results = hands.process(imgRGB)
    # pose_results = pose.process(imgRGB)
    # face_results = faceMesh.process(imgRGB)

    if rawImg is None:
        blackImg = generate_empty_image(tgh, tgw)
    else:
        print(rawImg.shape)
        blackImg = generate_empty_image(rawImg.shape[0], rawImg.shape[1])

    if handsOn and hand_results.multi_hand_landmarks:
        for hid, handLms in enumerate(hand_results.multi_hand_landmarks):
            mpDraw.draw_landmarks(blackImg, handLms, mpHands.HAND_CONNECTIONS, 
                                  landmark_drawing_spec=drawing_spec, 
                                  connection_drawing_spec=connection_drawing_spec)

    if poseOn and pose_results.pose_landmarks:
        mpDraw.draw_landmarks(blackImg, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS, 
                              landmark_drawing_spec=drawing_spec, 
                              connection_drawing_spec=connection_drawing_spec)

    if faceOn and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, 
                                  connections=mpFaceMesh.FACEMESH_CONTOURS, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(blackImg, landmark_list=face_landmarks, 
                                  connections=mpFaceMesh.FACEMESH_TESSELATION, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    return blackImg

### On Original Raw Image
def findHandsOnOriginal(results, rawImg, tgh=224, tgw=224, flipType=True, trackHands=False, bboxVisible=True, black=False):
    if results is None:
        imgRGB = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(imgRGB)
    else:
        hand_results = results

    allHands = []
    h, w, c = rawImg.shape

    blackImg = generate_empty_image(h, w)

    if hand_results.multi_hand_landmarks:
        for handType, handLms in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
            myHand = {}
            ## lmList
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)

            ## bbox
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2),bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            if flipType:
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"
            else:
                myHand["type"] = handType.classification[0].label
            allHands.append(myHand)

            ## draw
            if trackHands:
                mpDraw.draw_landmarks(rawImg, handLms, mpHands.HAND_CONNECTIONS)
                cv2.rectangle(rawImg, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 1)
                cv2.putText(rawImg, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
                if black:
                    mpDraw.draw_landmarks(blackImg, handLms, mpHands.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec, 
                                                                connection_drawing_spec=connection_drawing_spec)
                    # cv2.rectangle(blackImg, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 1)
                    # cv2.putText(blackImg, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    return allHands, rawImg, blackImg

### On Black image
def findHandsOnBlack(rawImg, tgh=224, tgw=224, flipType=True, bboxOn=False, bboxVisible=True):

    imgRGB = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(imgRGB)

    allHands = []
    h, w, c = rawImg.shape

    # blackImg = generate_empty_image(tgh, tgw)

    if hand_results.multi_hand_landmarks:
        for handType, handLms in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
            myHand = {}
            ## lmList
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                # px, py, pz = int(lm.x * tgw), int(lm.y * tgh), int(lm.z * tgw)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)

            ## bbox
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2),bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            if flipType:
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"
            else:
                myHand["type"] = handType.classification[0].label
            allHands.append(myHand)

            ## draw
            if bboxOn:
                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                cv2.rectangle(rawImg, (bbox[0] - b_offset, bbox[1] - b_offset), (bbox[0] + bbox[2] + b_offset, bbox[1] + bbox[3] + b_offset), (255, 0, 255), 1)
                cv2.putText(rawImg, myHand["type"], (bbox[0] - type_offset, bbox[1] - type_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    return allHands, rawImg
