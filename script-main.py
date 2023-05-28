import os
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
# import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# from tensorflow.keras import layers
# from tensorflow.keras import Model
# from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import set_session
from flask import Flask, request
from flask_cors import CORS
import cv2
import urllib
import urllib.request
import json
import numpy as np
import base64
from datetime import datetime
# from keras_facenet import FaceNet





# face verification with the VGGFace2 model
# check version of keras_vggface
import keras_vggface
import mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import requests
from io import BytesIO



#eye blink part imports
import cv2 # for video rendering
import dlib # for face and landmark detection
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils



graph = tf.compat.v1.get_default_graph()
app = Flask(__name__)
CORS(app)
sess = tf.compat.v1.Session()
set_session(sess)




def calculate_EAR(eye):

	# calculate the vertical distances
	y1 = dist.euclidean(eye[1], eye[5])
	y2 = dist.euclidean(eye[2], eye[4])

	# calculate the horizontal distance
	x1 = dist.euclidean(eye[0], eye[3])

	# calculate the EAR
	EAR = (y1+y2) / x1
	return EAR


def isBlink(filename):
    # Variables
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0


    # Eye landmarks
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    # Initializing the Models for Landmark and
    # face Detection
    detector = dlib.get_frontal_face_detector()
    landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    print(filename)
    img = cv2.imread(filename)
    #img = np.asarray(img)
    #print("this is the image of the blink test")
    #print(img)
    frame = imutils.resize(img, width=640)

    # converting frame to gray scale to
    # pass to detector
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = detector(img_gray)
    for face in faces:

        # landmark detection
        shape = landmark_predict(img_gray, face)

        # converting the shape class directly
        # to a list of (x,y) coordinates
        shape = face_utils.shape_to_np(shape)

        # parsing the landmarks list to extract
        # lefteye and righteye landmarks--#
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]

        # Calculate the EAR
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)

        # Avg of left and right eye EAR
        avg = (left_EAR+right_EAR)/2
        if avg < blink_thresh:
            print("BLINK DETECTED")
            cv2.imwrite('images/blink.jpg', img)
            return 'true'
        else:
            print("BLINK NOT DETECTED")
            return 'false'
    return 'face not detected'




# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
#    pixels = np.expand_dims(pixels, axis=0)
    
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings_normie(filenames):
    # extract faces
#    face = extract_face(filename)
    face = [extract_face(f) for f in filenames]
    # convert into an array of samples
    sample = asarray(face, 'float32')
    # prepare the face for the model, e.g. center pixels
    sample = preprocess_input(sample, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(sample)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return 'true'
    else:
        return 'false'

    
# Our Flask Server

@app.route('/verify', methods=['POST'])
def verifyIdentity():
    
    img_data = request.get_json()['image64']

#    url = request.get_json()['imgUrl']
    ID = request.get_json()['ID']
    imgPath = 'images/'+ID+'.jpeg'
#    urls = [url,url,url,url]
    urls = [imgPath,imgPath,imgPath,imgPath]
    img_name = str(int(datetime.timestamp(datetime.now())))
    with open('images/'+img_name+'.jpg', "wb") as fh:
        fh.write(base64.b64decode(img_data[22:]))
    path = 'images/' + img_name +'.jpg'
    paths = [path,path,path,path]

    blinkPath = 'images/blink.jpg'
    blinkPaths = [blinkPath, blinkPath, blinkPath, blinkPath]

    if not os.path.isfile(blinkPath):
        return json.dumps({"identified": 'false'})
    else:
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            curr_embedding = get_embeddings_normie(paths)
            original_embedding = get_embeddings_normie(urls)
            blink_embedding = get_embeddings_normie(blinkPaths)
            matched = is_match(curr_embedding[0], original_embedding[0])
            matched2 = is_match(blink_embedding[0], original_embedding[0])

            if matched == 'false' or matched2 == 'false':
                matched = 'false'
        os.remove(path)
        os.remove(blinkPath)
        return json.dumps({"identified": matched})



@app.route('/checkBlink', methods=['POST'])
def catchBlink():
    img_data_blink = request.get_json()['image64']
    blinkPath = 'images/blinkTest.jpg'
    with open(blinkPath, "wb") as fh:
        fh.write(base64.b64decode(img_data_blink[22:]))

    blink = isBlink(blinkPath)
    return json.dumps({"blink": blink})


