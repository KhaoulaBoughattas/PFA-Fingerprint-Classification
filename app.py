import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os

# -------------------
# Configuration
# -------------------
MODEL_PATH = "model_fingerprint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Fake", "Live"]

# -------------------
# Chargement du modèle
# -------------------
def load_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------
# Prétraitement de l'image
# -------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# -------------------
# Prédiction
# -------------------
def predict(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return CLASS_NAMES[predicted.item()]

# -------------------
# Interface Streamlit
# -------------------
st.set_page_config(page_title="Fingerprint Classifier", page_icon="🌐", layout="centered")
st.title("🔍 Vers une authentification fiable")
st.subheader("🌐 Système de détection d'empreintes digitales : Réelles vs Falsifiées")

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .title, .subtitle, .stButton>button {
        font-family: 'Arial';
    }
    .stButton>button {
        background-color: #ffc107;
        color: black;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" Importer une empreinte digitale (.png ou .bmp)", type=["png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Empreinte chargée", use_column_width=True)

    if st.button("Analyser l'image"):
        with st.spinner("🪄 Prédiction en cours..."):
            model = load_model()
            result = predict(model, image)
        st.success(f"Résultat : **{result}** 🌟")
        if result == "Live":
            st.balloons()
        else:
            st.warning("Empreinte suspecte d'être falsifiée 🚫")

st.markdown("---")
st.caption("Made with ❤️ by Khaoula Boughattas | ENET'COM")
