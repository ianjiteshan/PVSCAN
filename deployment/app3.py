# PVSCAN - AI-based Solar Panel Health Analyzer
# Streamlit App with Toggleable Dark/Light Theme, Single + Batch Image Classification

import streamlit as st
from PIL import Image
import zipfile
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import tempfile
import pandas as pd

# -------------------- CONFIG -------------------- #

APP_NAME = "PVSCAN"
DESCRIPTION = "AI-based Solar Panel Health Analyzer"
LOGO_COMP = "logo_comp.png"
LOGO_PHONE = "logo_phone.png"

MODEL_PATHS = {
    "SPICE AI v1.1": "spice_ai_mobilenetv3_v1.1.pth",
    "SPICE AI v2.0": "spice_ai_mobilenetv3_v2.0.pth"
}

CLASS_LABELS = [
    "Clean", "Dust", "Snow", "Shadow",
    "Bird Dropping", "Crack", "Discoloration", "Hotspot"
]

CLASS_CONFIG = {
    "Clean": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Panel is clean and functioning well."},
    "Dust": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Dust detected. Cleaning recommended."},
    "Snow": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Snow detected. Remove for optimal output."},
    "Shadow": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Shading issue. Check panel placement."},
    "Bird Dropping": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Bird droppings affecting performance. Clean required."},
    "Crack": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Physical damage detected. Consider replacement."},
    "Discoloration": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Discoloration seen. May affect output."},
    "Hotspot": {"min_score": 0.5, "max_score": 1.0, "suggestion": "Hotspot detected. Inspect panel for faults."}
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- MODEL UTILS -------------------- #

def load_model(model_path):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 960),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(960, 8)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_suggestions(label, score):
    config = CLASS_CONFIG.get(label, {})
    if config and config['min_score'] <= score <= config['max_score']:
        return config['suggestion']
    return "No specific suggestion."

# -------------------- CLASSIFICATION UTILS -------------------- #

def classify_image(model, image):
    img_tensor = transform(image).unsqueeze(0)
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    score, pred = torch.max(probs, 1)
    label = CLASS_LABELS[pred.item()]
    return label, score.item(), generate_suggestions(label, score.item())

# -------------------- MAIN APP -------------------- #

def app_ui():
    st.set_page_config(page_title=APP_NAME, layout="centered")

    st.markdown(f"""
    <style>
    .main {{ background-color: {'#0e1117' if st.session_state.theme == 'Dark' else '#ffffff'}; color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'}; }}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.image(LOGO_COMP, use_column_width=True)
    st.sidebar.title(APP_NAME)
    st.sidebar.caption(DESCRIPTION)

    page = st.sidebar.selectbox("Choose Mode", ["Single Image", "Batch Analysis"])
    model_choice = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
    st.sidebar.button("Toggle Theme", on_click=toggle_theme)

    model = load_model(MODEL_PATHS[model_choice])

    st.title(APP_NAME)
    st.caption(DESCRIPTION)

    if page == "Single Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            label, score, suggestion = classify_image(model, image)
            st.success(f"Prediction: {label} ({score:.2f})")
            st.info(suggestion)

    elif page == "Batch Analysis":
        uploaded_zip = st.file_uploader("Upload a ZIP file of images", type="zip")
        if uploaded_zip:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_ref.extractall(tmpdirname)
                    results = []
                    for file in os.listdir(tmpdirname):
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            img_path = os.path.join(tmpdirname, file)
                            image = Image.open(img_path).convert('RGB')
                            label, score, suggestion = classify_image(model, image)
                            results.append({"Image": file, "Prediction": label, "Confidence": f"{score:.2f}", "Suggestion": suggestion})
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results as CSV", csv, "batch_results.csv", "text/csv")

# -------------------- THEME TOGGLER -------------------- #

def toggle_theme():
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"

# Initialize theme state
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Run the app
app_ui()
