import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Aquarium Detection", page_icon="🐠")

st.title("🐠 Aquarium Object Detection")
st.write("ตรวจจับสัตว์น้ำ 7 ชนิด")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

confidence = st.slider(
    "ปรับระดับความมั่นใจ (Confidence)",
    0.0, 1.0, 0.25, 0.05
)

st.write(f"ความมั่นใจขั้นต่ำ: {confidence * 100:.0f}%")

uploaded_file = st.file_uploader(
    "เลือกรูปภาพ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(image, conf=confidence)

    st.image(results[0].plot(), caption="ผลการตรวจจับ")

    st.subheader("รายละเอียดผลการตรวจจับ")

    if len(results[0].boxes) == 0:
        st.warning("ไม่พบสัตว์น้ำในภาพ")
    else:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0]) * 100
            name = model.names[class_id]

            st.write(f"{name} — {score:.2f}%")
