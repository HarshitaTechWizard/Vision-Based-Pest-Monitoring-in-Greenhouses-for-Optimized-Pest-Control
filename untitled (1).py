import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Path to your saved Keras model (.keras or .h5 format)
MODEL_PATH = "ip02_pest_model.h5"
CLASSES_FILE = "classes.txt"
IMG_SIZE = (224, 224)

# Cache model and class names for performance
@st.cache(allow_output_mutation=True)
def load_model_and_classes():
    # Load the full model (architecture + weights) from .keras or .h5
    model = tf.keras.models.load_model(MODEL_PATH)
    # Load class names
    with open(CLASSES_FILE, "r") as f:
        class_names = [line.strip() for line in f]
    return model, class_names

model, class_names = load_model_and_classes()

st.title("Greenhouse Pest Monitoring App")
st.write("Upload or select an image to detect the pest class.")

# Sidebar: Option to use test dataset sample
use_test = st.sidebar.checkbox("Use an example from test dataset")

if use_test:
    test_dir = os.path.join("classification", "test")
    class_dirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    sel_class = st.sidebar.selectbox("Select Class", class_dirs)
    img_files = sorted(os.listdir(os.path.join(test_dir, sel_class)))
    sel_img = st.sidebar.selectbox("Select Image", img_files)
    img_path = os.path.join(test_dir, sel_class, sel_img)
    image = Image.open(img_path).convert("RGB")
    st.image(image, caption=f"{sel_class}/{sel_img}", use_column_width=True)
else:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.warning("Please upload an image or select an example from the test dataset.")
        st.stop()
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Preprocess the image
resized = image.resize(IMG_SIZE)
array = np.array(resized)
input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(array)
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run prediction
preds = model.predict(input_tensor)[0]
top5 = preds.argsort()[-5:][::-1]

st.subheader("Top 5 Predictions")
for idx in top5:
    st.write(f"{class_names[idx]}: {preds[idx]*100:.2f}%")

st.markdown("---")
st.write(f"**Model file:** {MODEL_PATH}  |  **Classes file:** {CLASSES_FILE}")

