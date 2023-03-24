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



graph = tf.compat.v1.get_default_graph();
app = Flask(__name__)
CORS(app)
sess = tf.compat.v1.Session()
set_session(sess)


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

#def extract_face_outcast(path, required_size=(224, 224)):
#    # load image from file
#
##    response = requests.get(path)
##    pixels = Image.open(BytesIO(response.content))
## getting image from db
#
#
#    pixels = pyplot.imread(filename)
#
#    # create the detector, using default weights
#    detector = MTCNN()
#    # detect faces in the image
#    results = detector.detect_faces(pixels)
#    # extract the bounding box from the first face
#    x1, y1, width, height = results[0]['box']
#    x2, y2 = x1 + width, y1 + height
#    # extract the face
#    face = pixels[y1:y2, x1:x2]
#    # resize pixels to the model size
#    image = Image.fromarray(face)
#    image = image.resize(required_size)
#    face_array = asarray(image)
#    return face_array


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


#def get_embeddings_outcast(filename):
#    # extract faces
#    face = extract_face_outcast(filename)
#    # convert into an array of samples
#    sample = asarray(face, 'float32')
#    # prepare the face for the model, e.g. center pixels
#    sample = preprocess_input(sample, version=2)
#    # create a vggface model
#    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#    # perform prediction
#    yhat = model.predict(sample)
#    return yhat




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
    print(request.get_json());
    img_data = request.get_json()['image64']
    print(img_data)
#    url = request.get_json()['imgUrl']
    ID = request.get_json()['ID']
    imgPath = 'images/'+ID+'.jpeg';
#    urls = [url,url,url,url]
    urls = [imgPath,imgPath,imgPath,imgPath]
    img_name = str(int(datetime.timestamp(datetime.now())))
    with open('images/'+img_name+'.jpg', "wb") as fh:
        fh.write(base64.b64decode(img_data[22:]))
    path = 'images/' + img_name +'.jpg'
#    path = 'images/' + 'hello.jpg'
    paths = [path,path,path,path];
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        curr_embedding = get_embeddings_normie(paths)
        original_embedding = get_embeddings_normie(urls)
        matched = is_match(curr_embedding[0], original_embedding[0])
    os.remove(path)
    return json.dumps({"identified": matched})

    
