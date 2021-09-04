import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
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
from gen_mask import *
import keras.backend as K

# Declare a flask app
app = Flask(__name__)

# Class Labels for Output
class_labels = {0: 'AMD', 1: 'DR', 2: 'NORMAL'}
weights = [[0.1, 0.4, 0.5], [0.9, 0.6, 0.5]]

# Custom IOU metric for segmentation
def iou(y_true, y_pred):
    smooth = 1.
    intersection = K.sum(y_true * y_pred)
    summ = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (summ - intersection + smooth)
    return iou

# Aggregate for Ensemble Prediction
def ensemble_predictions(members, weights, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    # summed = np.sum(yhats, axis=0)
    summed = np.multiply(yhats[0], weights[0])
    for i in range(1, len(members)):
        summed += np.multiply(yhats[i], weights[i])
    # argmax across classes
    pred_proba = "{:.3f}".format(np.amax(summed))
    result = np.argmax(summed, axis=1)
    return int(result), pred_proba

#OCT MODEL
VGG_MODEL_PATH = 'https://www.kaggle.com/itsmariodias/pretrained-datasets?select=vgg_kermanyduketehran.h5'  #'models/vgg_kermany+duke+tehran.h5'
OPTICNET_MODEL_PATH ='https://www.kaggle.com/itsmariodias/pretrained-datasets?select=opticnet_kermanyduketehran.h5' #'models/opticnet_kermany+duke+tehran.h5'

# #FUNDUS MODEL
# FUNDUS_MODEL_PATH = 'models/opticnet_kermany+duke+tehran.h5'
# fundus_model = load_model(FUNDUS_MODEL_PATH)
# fundus_model.make_predict_function()

#SEGMENT MODEL
SEGMENT_MODEL_PATH = '.git/lfs/objects/1d/52/1d526fa0faeb9635a75b099e32f545298028f9966653365d9a6101d1cb82ee3d' #'models/unet_aroi.hdf5'
segment_model = load_model(SEGMENT_MODEL_PATH, custom_objects={'iou': iou})
segment_model.make_predict_function()

print('Models loaded. Start serving...')
choice = 0

members = [load_model(VGG_MODEL_PATH), load_model(OPTICNET_MODEL_PATH)]

def model_predict_oct(img, weights, members):
    img_resized = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img_resized)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    pred_class, proba = ensemble_predictions(members, weights, x)
    return pred_class, proba

# def model_predict_fundus(img, model):
#     img_resized = img.resize((224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img_resized)
#     x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     preds = model.predict(x)
#     return preds


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

# @app.route('/fundus', methods=['GET'])
# def fundus():
#     global choice
#     choice = 1
#     return render_template('fundus.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Predicting...")
    if request.method == 'POST':
        # For Fundus
        if choice==1:
            pass
            # # Get the image from post request
            # img = base64_to_pil(request.json)

            # # Save the image to ./uploads
            # # img.save("./uploads/image.png")

            # # Make prediction
            # preds = model_predict_oct(img, oct_model)

            # # Process your result for human
            # pred_proba = "{:.3f}".format(np.amax(preds))   # Max probability
            # pred_class = int(np.argmax(preds, axis = 1))   # ImageNet Decode

            # heatmap = generate_heatmap(img, FUNDUS_MODEL_PATH, 'Add4', 'Dense_2')
            # heatmap = pil_tobase64(heatmap)
            # # result = str(pred_class[0][0][1])               # Convert to string
            # # result = result.replace('_', ' ').capitalize()

            
            # # Serialize the result, you can add additional fields
            # return jsonify(result=class_labels[pred_class], probability=pred_proba, heatmap = heatmap)
        # For segmentation
        elif choice==2:
            # Get the image from post request
            #pass
            img = base64_to_pil(request.json)

            # Save the image to ./uploads
            # img.save("./uploads/image.png")

            # Make prediction
            # preds = model_predict_oct(img, oct_model)

            # Process your result for human
            # pred_proba = "{:.3f}".format(np.amax(preds))   # Max probability
            # pred_class = int(np.argmax(preds, axis = 1))   # ImageNet Decode

            mask = show(segment_model, img)
            mask = pil_tobase64(mask)
            # result = str(pred_class[0][0][1])               # Convert to string
            # result = result.replace('_', ' ').capitalize()

            
            # Serialize the result, you can add additional fields
            return jsonify(result='', probability='', heatmap = mask)
        # For OCT classification
        else:
            # Get the image from post request
            img = base64_to_pil(request.json)

            # Save the image to ./uploads
            # img.save("./uploads/image.png")

            # Make prediction
            pred_class, pred_proba = model_predict_oct(img, weights, members)

            heatmap = generate_heatmap(img, VGG_MODEL_PATH, 'block5_conv3', 'dense_23')
            heatmap = pil_tobase64(heatmap)
            # result = str(pred_class[0][0][1])               # Convert to string
            # result = result.replace('_', ' ').capitalize()

            
            # Serialize the result, you can add additional fields
            return jsonify(result=class_labels[pred_class], probability=pred_proba, heatmap = heatmap)

    return None


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=33507, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0',33507), app)
    http_server.serve_forever()
"""
    app.listen(process.env.PORT || '5002', function(){
  console.log("Express server listening on port %d in %s mode", this.address().port, app.settings.env);
});
"""
