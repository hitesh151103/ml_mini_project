import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
model = load_model('fish_classifier_model.h5') 
class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']
st.title("🐟 Fish Classifier")
st.write("Upload a fish image to classify its species.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    predicted_index = np.argmax(prediction)
    confidence_score = prediction[0][predicted_index] * 100

    st.write(f"### 🐠 Predicted Fish: `{predicted_class}`")
    st.write(f"### 🔍 Confidence: `{confidence_score:.2f}%`")

