import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model



def streamlit():
    
    classifier_model = "cnn_sign.h5"
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    try:
        file = st.sidebar.file_uploader(label = 'Upload an image', type = ["png","jpg","jpeg"] )
        class_btn = st.sidebar.button("Classify")
        
        
        
        if file is not None: 
                
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=None)
            
            basewidth = 28
            img = Image.open(file).convert('L')
            width_percent = (basewidth / float(img.size[0]))
            hight_size = int((float(img.size[1]) * float(width_percent)))
            img = img.resize((basewidth, hight_size), Image.ANTIALIAS)

            img_np = np.array(img) / 255 # noramlize
            img_np.reshape(1, 28, 28, 1)
            img_np = np.array([img_np]) # add dimesion
            
            if class_btn == True:
                prediction = model.predict(img_np)
                pred_list = list(prediction)
                pred_list = [round(pred_list[0][i]) for i in range(24) ]
                pred = pred_list.index(1)
                st.write("Prediction of image is :")
                st.write(labels[pred])
                st.success('Classified')
        if file is  None: 
                st.title('sing language Classifier')

                st.subheader("Welcome to this simple web application that classifies sing language. The sing language  classified into 24 different classes namely: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.")
        else:
            pass        
    except:
        
        st.warning("please upload a valid image file format")
        st.info(" Rerun or press the   R  key")
   
    
streamlit()

            