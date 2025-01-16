import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#load_model
model=tf.keras.models.load_model('cifar10_cnn_model.h5')

#model labels
labels=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

def process_image(image):
    image=image.resize((32,32))
    image=np.array(image)/255.0
    image=np.expand_dims(image,axis=0)
    return image
def predict(image):
    processed_image=process_image(image)
    predictions=model.predict(processed_image)
    return labels[np.argmax(predictions)]
#streamlit app
st.title("Image Prediction with the CIFAR-10 Model")
st.write("Upload an image to make some predictions with the model.")

uploaded_file=st.file_uploader("Choose an image for upload",type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded_image",use_column_width=True)

    st.write("Predicting")
    prediction=predict(image)
    st.write(f"Prediction: {prediction}")