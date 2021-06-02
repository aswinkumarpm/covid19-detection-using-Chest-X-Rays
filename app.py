from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Detection_Covid_19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
graph = tf.get_default_graph()

print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('test')
print('Model loaded. Check http://127.0.0.1:5000/')





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']
            print(f, "hgahshgsghasghgs")

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(img_path=file_path, model=model)
            print(file_path, model)
            
            if preds[0][0] == 0:
                prediction = 'Positive For Covid-19'
            else:
                prediction = 'Negative for Covid-19'
            # Process your result for human
            # pred_class = preds.argmax(axis=-1)            # Simple argmax
            # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
            # result = str(pred_class[0][0][1)               # Convert to string
            return prediction
        return None


def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict_classes(xtest_image)
    print(preds)
    return preds

if __name__ == '__main__':
    app.run(debug=True)

