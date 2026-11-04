import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Aquarium Detection", page_icon="🐠")

st.title("🐠 Aquarium Object Detection")
st.write("อัปโหลดรูปภาพเพื่อตรวจจับสัตว์น้ำ")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "เลือกรูปภาพ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(image)

    result_image = results[0].plot()

    st.image(result_image, caption="ผลการตรวจจับ")