import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Helmet / No Helmet Detection")

@st.cache_resource
def load_model():
    return YOLO("best.pt")   # YOLO 11n model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(image)

    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)
