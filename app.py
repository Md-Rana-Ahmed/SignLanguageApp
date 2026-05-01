import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.h5")

class_names = ["A","B","C","D","E","F","G","H","I","J"]

st.title("Sign Language to Text AI System")

uploaded_file = st.file_uploader("Upload Hand Image", type=["jpg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = class_names[np.argmax(pred)]

    st.success("Prediction: " + result)