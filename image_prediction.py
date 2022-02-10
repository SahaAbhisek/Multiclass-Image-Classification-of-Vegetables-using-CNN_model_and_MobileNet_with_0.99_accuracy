#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:34:05 2022

@author: abhisek
"""

import streamlit as st
import tensorflow as tf
tf.__version__

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Sequential
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

st.header("Vegitable Classifier Predictor")

def main():
    file_uploaded = st.file_uploader("Please upload your vegetable image", type=['png','jpg','jpeg'])
    
    if file_uploaded is not None:
        img = load_img(file_uploaded, target_size=(150,150,3))
        i = img_to_array(img)/255.
        img_arr = np.array(i)
        
        figure = plt.figure()
        plt.imshow(img_arr)
        plt.axis('off')
        
        result = predict_class(img_arr)
        st.write(result)
        st.pyplot(figure)
        
def predict_class(img_arr):
    Classifier = load_model('/content/drive/MyDrive/Colab Notebooks/Vegetable_classification/vegetable_classifier_mnet.h5')
    shape = (150,150,3)
    model = Sequential(hub[hub.KerasLayer(Classifier, input_shape=shape)])
    prediction = model.predict(img_arr.reshape(-1, 150, 150, 3))
    class_list = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 
                  'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya',
                  'Potato', 'Pumpkin', 'Radish', 'Tomato']
    
    pred = class_list[np.argmax(prediction)]
    result = "The image uploaded is a: {}".format(pred)
    
    return result

if __name__=='__main__':
    main()
    
        