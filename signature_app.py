import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

# Load models
@st.cache_resource
def load_models():
    vgg_model = load_model("models/vgg_model.h5")
    mobilenet_model = load_model("models/mobilenet_model.h5")
    return vgg_model, mobilenet_model

# Preprocess function
def preprocess(image_file):
    img = Image.open(image_file).convert("L")
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.stack((img,) * 3, axis=-1)  # Convert to 3-channel
    return np.expand_dims(img, axis=0)

# Predict using a model
def predict(model, real_img, test_img, name):
    diff = np.abs(real_img - test_img)
    pred = model.predict([real_img, test_img])[0][0]
    return {
        "model": name,
        "prediction": "Genuine" if pred < 0.5 else "Forged",
        "accuracy": round((1 - pred) * 100, 2)
    }

# Streamlit UI
st.title("🖊️ Signature Verification System")
st.markdown("Upload two signatures (one real, one test) to verify using VGG16 and MobileNetV2 models.")

real_signature = st.file_uploader("Upload Real Signature", type=["jpg", "png", "jpeg"])
test_signature = st.file_uploader("Upload Test Signature", type=["jpg", "png", "jpeg"])

if real_signature and test_signature:
    vgg_model, mobilenet_model = load_models()
    
    real_img = preprocess(real_signature)
    test_img = preprocess(test_signature)

    st.image([real_signature, test_signature], caption=["Real Signature", "Test Signature"], width=200)

    result_vgg = predict(vgg_model, real_img, test_img, "VGG16")
    result_mobilenet = predict(mobilenet_model, real_img, test_img, "MobileNetV2")

    st.subheader("🔍 Verification Results")
    
    for result in [result_vgg, result_mobilenet]:
        st.markdown(f"""
        ### Model: {result['model']}
        - **Prediction**: {result['prediction']}
        - **Accuracy**: {result['accuracy']}%
        """)

