from PIL import Image
import numpy as np 
import streamlit as st 
from keras.models import load_model
import keras.utils as image
import os
import cv2
import tensorflow as tf

IMAGE_MODEL = 'artifacts/model/emergency_words_model_v1.hdf5'
VIDEO_MODEL = 'artifacts/model/emergency_words_model_v1.hdf5'

video_model = load_model(VIDEO_MODEL)
video_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

emergency_words = ['Accident', 'Call', 'Doctor', 'Help', 'Hot', 'Lose', 'Pain', 'Thief']


def predictImage(fame):
    print("Image prediction .....")

def predictVideo(frameSet):
    print("Video prediction .....")

def predictLiveStream(frameSet):
    # print(frameSet.shape)
    result = None
    if frameSet.shape[0] != 0:
        print(frameSet.shape)
        frameSet = tf.expand_dims(frameSet, axis=0)
        print(frameSet.shape)
        # model = load_model(VIDEO_MODEL)
        # result = video_model.predict(frameSet)
        # print(result)
    return result




# def generateVideo():
#     img_array = []
#     for filename in glob.glob('C:/New folder/Images/*.jpg'):
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width,height)
#         img_array.append(img)
    
    
#         out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
#         for i in range(len(img_array)):
#             out.write(img_array[i])
#         out.release()










# def process_image(image_name):
#     #print('Process image: ', image_name)
#     model = load_model('\artifacts\model\emergency_words_model_v1.hdf5')
#     img = image.load_img(image_name, target_size = (28,28))
#     # convert image into array for prediction
#     test_image = image.img_to_array(img)
#     test_image = np.expand_dims(test_image, axis = 0)
#     # predict image using model
#     result = model.predict(test_image).argmax()

# def predictLiveStream(frameSet):
#     model = load_model('\artifacts\model\emergency_words_model_v1.hdf5')
#     result = model.predict(frameSet).argmax()

# def generateFramesSequence(frame):
#     frameSet = []
#     frame_counter = 0
#     while(True):

#         frame_counter +=1

# def generateVideo():
#     img_array = []
#     for filename in glob.glob('C:/New folder/Images/*.jpg'):
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width,height)
#         img_array.append(img)
    
    
#         out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
#         for i in range(len(img_array)):
#             out.write(img_array[i])
#         out.release()