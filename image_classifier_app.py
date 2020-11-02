import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps 


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('image_classifier_model.hdf5')
  return model
model = load_model()

st.write("""
          # My ML Neural Network Image Classifier
         """)

file = st.file_uploader("Please upload an image (jpg or png). Image should be a building, \
                  forest, glacier, mountain, sea, or street. For more information, please visit:\
                  https://github.com/ngidingidi/nn-image-classifier-streamlit ", 
                         type=["jpg", "png"])

def import_and_predict(image_data, model):
  size = (180,180)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS) 
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  prediction = model.predict(img_reshape)

  return prediction

if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  predictions = import_and_predict(image, model)
  class_names=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
  string="The most likely class of this beautiful image is: " + class_names[np.argmax(predictions)]
  st.success(string)