import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load dari .h5
model = load_model("utils/model_suku.h5", compile=False)

detector = MTCNN()

# Label index
label_to_index = {
    'Jawa': 0,
    'Sunda': 1,
    'Cina': 2,
    'Minahasa': 3,
    'Betawi': 4
}
index_to_label = {v: k for k, v in label_to_index.items()}

# Fungsi prediksi wajah
def prediksi_wajah(image_pil):
    img_rgb = np.array(image_pil)
    faces = detector.detect_faces(img_rgb)

    if faces:
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = img_rgb[y:y+height, x:x+width]

        input_face = cv2.resize(cropped_face, (224, 224))
        input_face = input_face / 255.0
        input_face = np.expand_dims(input_face, axis=0)

        pred = model.predict(input_face)[0]
        kelas = np.argmax(pred)
        confidence = pred[kelas] * 100
        label = f"{index_to_label[kelas]} ({confidence:.2f}%)"

        return cropped_face, label
    else:
        return None, "Wajah tidak terdeteksi"

# Streamlit UI
st.set_page_config(page_title="Etnis Wajah", page_icon="🧑", layout="centered")

st.title("Deteksi Etnis Berdasarkan Wajah")

uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    with st.spinner("Mendeteksi wajah..."):
        face_img, pred_label = prediksi_wajah(image)

        if face_img is not None:
            st.markdown("### Wajah yang Terdeteksi")
            st.image(face_img, caption="Wajah yang Dipotong", use_column_width=False)
        else:
            st.warning("Wajah tidak terdeteksi")

        st.markdown(f"### Prediksi Etnis: **{pred_label}**")
