# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:59:06 2021

@author: User
"""
import os
import streamlit as st 
from PIL import Image
import numpy as np
from io import StringIO
#Import all important packages & libraries
import numpy as np
import pandas as pd
import sklearn
import re
from glob import glob
import xml.etree.ElementTree as ET
#import glob
import os
import json

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
import itertools
import cv2
import random
import datetime
import shutil
#import wget
random.seed(1)
import PIL
from PIL import Image, ImageDraw
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mp
import seaborn as sns

#Import sklearn libraries
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelBinarizer

#Import tensorflow libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,Convolution2D,MaxPooling2D,MaxPool2D,Activation,GlobalMaxPool2D,GlobalAveragePooling2D,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Concatenate, UpSampling2D, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import binary_crossentropy, CategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model

#Import keras libraries
from keras.applications import xception
from keras import utils
from keras.utils.np_utils import to_categorical  
from keras.utils import np_utils

from tqdm import tqdm

# Print versions
print(f'Pandas version: {pd.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'Tensorflow version: {tf.__version__}')
print(f'CV version: {cv2.__version__}')


def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by <a style='display: block; text-align: center;' >Shubhaditya Goswami</a></p>
    </div>
    """
    return footer


def app():
    """
    Main function that contains the application for getting predictions from 
    keras based trained models.
    """
    # Get list of saved hdf5 models, which will be displayed in option to load.
    hdf5_file_list = [file for file in os.listdir("./model") if file.endswith(".hdf5")]
    hdf5_file_names = [os.path.splitext(file)[0] for file in hdf5_file_list]
    
    st.title("Keras Prediction Basic UI")
    st.header("A Streamlit based Web UI To Get Predictions From Trained Models")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    model_type = st.radio("Choose trained model to load...", hdf5_file_names)
    
    loaded_model = tf.keras.models.load_model("./model/{}.hdf5".format(model_type))
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        if "mnist" in model_type:
            image = Image.open(uploaded_file)
            image = image.resize((28,28), Image.NEAREST)
            st.image(image, caption='Uploaded Image.', use_column_width=False)
            st.write("")
            st.write("Identifying...")
            # Convert to grayscale if RGB.
            print(image.size)
            print(image.mode)
            if image.mode == "RGB":
                image = image.convert("L")
            # Convert to numpy array and resize.
            image = np.array(image)
            image = np.resize(image,(1,784))
            
            # Get prediction.
            yhat = loaded_model.predict(image)
            # Convert the probabilities to class labels
            label = np.argmax(yhat, axis=1)[0]
            st.write('%s' % (label) )
            

if __name__=='__main__':
    app()