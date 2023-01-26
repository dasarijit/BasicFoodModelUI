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

#Import tensorflow libraries
import tensorflow as tf

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
    <p>Developed by <a style='display: block; text-align: center;' >Group 5 - AIML</a></p>
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
    
    st.title("Food Prediction Basic UI")
    st.header("A Streamlit based Web UI To Get Predictions From Trained Models")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    model_type = st.radio("Choose trained model to load...", hdf5_file_names)
    
    loaded_model = tf.keras.models.load_model("./model/{}.hdf5".format(model_type),compile=False)
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        #if "mnist" in model_type:
            image = Image.open(uploaded_file)
         #   image = image.resize((28,28), Image.NEAREST)
            st.image(image, caption='Uploaded Image.', use_column_width=False)
            st.write("")
            st.write("Identifying...")
            # Convert to grayscale if RGB.
            print(image.size)
            print(image.mode)
          #  if image.mode == "RGB":
           #     image = image.convert("L")
            # Convert to numpy array and resize.
            #image = np.array(image)
            #image = np.resize(image,(1,784))
            
            # Get prediction.
            yhat = loaded_model.predict_class((loaded_model),(image), True)
            # Convert the probabilities to class labels
            label = np.argmax(yhat, axis=1)[0]
            st.write('%s' % (label) )
            

if __name__=='__main__':
    app()