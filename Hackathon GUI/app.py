import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil, pil_tobase64

from gen_heatmap import *


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()

#OCT MODEL
OCT_MODEL_PATH = 'models/vgg16.h5'
oct_model = load_model(OCT_MODEL_PATH)
oct_model.make_predict_function()
class_labels_oct = {0: 'CNV', 1: 'DME', 2: 'NORMAL'}

#FUNDUS MODEL
FUNDUS_MODEL_PATH = 'models/vgg16.h5'
fundus_model = load_model(FUNDUS_MODEL_PATH)
fundus_model.make_predict_function()
class_labels_fundus = {0: 'CNV', 1: 'DME', 2: 'NORMAL'}

#SEGMENT MODEL
SEGMENT_MODEL_PATH = 'models/vgg16.h5'
segment_model = load_model(SEGMENT_MODEL_PATH)
segment_model.make_predict_function()
class_labels_segment = {0: 'CNV', 1: 'DME', 2: 'NORMAL'}


print('Models loaded. Start serving...')
choice = 0

def model_predict_oct(img, model):
    img_resized = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img_resized)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds

def model_predict_fundus(img, model):
    img_resized = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img_resized)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    global choice
    choice = 0
    return render_template('index.html')

@app.route('/segmentation', methods=['GET'])
def segment():
    global choice
    choice = 2
    return render_template('segment.html')

@app.route('/fundus', methods=['GET'])
def fundus():
    global choice
    choice = 1
    return render_template('fundus.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Predicting...")
    if request.method == 'POST':
        # For Fundus
        if choice==1:
            pass
        # For segmentation
        elif choice==2:
            pass
        # For OCT classification
        else:
            # Get the image from post request
            img = base64_to_pil(request.json)

            # Save the image to ./uploads
            # img.save("./uploads/image.png")

            # Make prediction
            preds = model_predict_oct(img, oct_model)

            # Process your result for human
            pred_proba = "{:.3f}".format(np.amax(preds))   # Max probability
            pred_class = int(np.argmax(preds, axis = 1))   # ImageNet Decode

            heatmap = generate_heatmap(img, OCT_MODEL_PATH, 'block5_conv3')
            heatmap = pil_tobase64(heatmap)
            # result = str(pred_class[0][0][1])               # Convert to string
            # result = result.replace('_', ' ').capitalize()

            
            # Serialize the result, you can add additional fields
            return jsonify(result=class_labels_oct[pred_class], probability=pred_proba, heatmap = heatmap)

    return None


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
