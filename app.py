import streamlit as st
import numpy as np
import cv2
import keras
from PIL import Image

@st.cache_resource
def load_model():
    return keras.models.load_model('face_alignment_model.keras')

model = load_model()
TARGET_SIZE = 256

st.title("Facial Landmark Detection CNN")
st.write("Upload an image to see the neural network predict 5 facial landmarks in real-time.")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_image = np.array(image)
    orig_h, orig_w = input_image.shape[:2]

    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE))

    normalized = (resized / 255.0).astype(np.float32)
    input_tensor = np.expand_dims(normalized, axis=(0, -1))

    with st.spinner('Predicting landmarks...'):
        preds = model(input_tensor, training=False).numpy()
        pts = preds.reshape(-1, 2)

    scale_x = orig_w / TARGET_SIZE
    scale_y = orig_h / TARGET_SIZE

    output_img = input_image.copy()
    for (x, y) in pts:
        cx = int(x * scale_x)
        cy = int(y * scale_y)
        cv2.circle(output_img, (cx, cy), radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(output_img, (cx, cy), radius=12, color=(255, 255, 255), thickness=2)

    st.image(output_img, caption="Predicted Landmarks", use_container_width=True)