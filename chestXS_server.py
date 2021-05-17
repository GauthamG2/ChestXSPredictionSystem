from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from flask import Flask, url_for, send_from_directory, request, render_template
import logging
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from chestXS_test_deploy import testfunction, mainf

from symptomstodiseaseprediction import verify

import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers


# initialize flask
app = Flask(__name__, template_folder='templates')

# initialize server file
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# setting path directory
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# routing to home page
@app.route('/')
def index():
    return render_template('landing_page.html')

# create folder in local directory tp store images
def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

# getting values from the user input and passing to model

@app.route('/upload', methods=['POST'])

def upload():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        symptom_one = request.form['symptom_one']
        symptom_two = request.form['symptom_two']
        symptom_three = request.form['symptom_three']
        symptom_four = request.form['symptom_four']
        symptom_five = request.form['symptom_five']
        disease = verify(symptom_one, symptom_two, symptom_three, symptom_four, symptom_five)
        print("<<<<<<<<<<<<<<<<<< Disease - From Symptom -  First Print >>>>>>>>>>>>>>>>")
        print(disease)
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        
        img.save(saved_path)
        signs = mainf(saved_path)

        print("<<<<<<<<<<<<<<<<<<  Disease - From Image  >>>>>>>>>>>>>>>>")
        print(signs)
        print("<<<<<<<<<<<<<<<<<< Disease - From Symptom >>>>>>>>>>>>>>>>")
        print(disease)

        convertList = ''.join([str(e) for e in disease])
        print(convertList)

        if signs in convertList:
            print("IF EXECUTED")

            output = signs
            copd = "COPD or Emphysema or Chronicbronchitis"

            if signs == "COPDEmphysemaChronicbronchitis":
                output = copd
                return render_template('landing_page.html', output=output)
            else:
                return render_template('landing_page.html', output=output)
        else:
            print("ELSE EXECUTED")
            disease_not_predicted = "Disease could not be predicted!"
            return render_template('landing_page.html', output=disease_not_predicted)
    else:
        return render_template('landing_page.html')

@app.route('/moreInfo')
def moreInfo():
    return render_template('moreInfo.html')

class Config:

    def __init__(self):

        # Print the process or not
        self.verbose = True

        # Name of base network
        self.network = 'vgg'

        # Setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        # Anchor box scales
    # Note that if im_size is smaller, anchor_box_scales should be scaled
    # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = [64, 128, 256]

        # Anchor box ratios
        self.anchor_box_ratios = [
            [1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 300

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None

        self.model_path = None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)