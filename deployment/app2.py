import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import zipfile
import io
from collections import OrderedDict

# ---------------- CONFIG ----------------
CLASS_CONFIG = {
    "Panel Detected": {
        "min_score": 0.5,
        "max_score": 1.0,
        "suggestion": "Panel detected successfully."
    },
    "Clean Panel": {
        "min_score": 0.7,
        "max_score": 1.0,
        "suggestion": "Panel appears clean and functional."
    },
    "Physical Damage": {
        "min_score": 0.2,
        "max_score": 1.0,
        "suggestion": "Panel may be physically damaged. Consider inspection."
    },
    "Electrical Damage": {
        "min_score": 0.2,
        "max_score": 1.0,
        "suggestion": "Possible electrical issue. Check connections/inverter."
    },
    "Snow Covered": {
        "min_score": 0.3,
        "max_score": 1.0,
        "suggestion": "Snow on panel. Clear off for better performance."
    },
    "Soiled": {
        "min_score": 0.3,
        "max_score": 1.0,
        "suggestion": "Panel appears dirty. Clean it for optimal efficiency."
    },
    "Shaded": {
        "min_score": 0.3,
        "max_score": 1.0,
        "suggestion": "Panel is shaded. Check for trees/buildings causing shadow."
    },
    "Bird Droppings": {
        "min_score": 0.2,
        "max_score": 1.0,
        "suggestion": "Droppings detected. Clean panel to improve output."
    }
}

MODEL_PATHS = {
    "SPICE AI v1.1": "spice_ai_mobilenetv3_v1.1.pth",
    "SPICE AI v2.0": "spice_ai_mobilenetv3_v2.0.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- LOADERS -----------------
@st.cache_resource
def load_model(model_path):
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, len(CLASS_CONFIG))
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        results = OrderedDict()
        for i, label in enumerate(CLASS_CONFIG.keys()):
            results[label] = round(float(probs[i]) * 100, 2)
        return results

def generate_suggestions(results):
    suggestions = []
    for label, score in results.items():
        config = CLASS_CONFIG[label]
        if config["min_score"] * 100 <= score <= config["max_score"] * 100:
            suggestions.append(f"🔹 **{label}**: {config['suggestion']} ({score}%)")
    return suggestions

# ----------------- UI -----------------
st.set_page_config(page_title="Solar PV Analyzer", layout="centered")
st.image("logo_comp.png", width=250)
st.image("logo_phone.png", width=150)

st.markdown("## ☀️ Solar PV Inference App")
st.markdown("Upload **a single image** or a **ZIP file** of images to analyze panel conditions using AI.")

col1, col2 = st.columns(2)
mode = col1.radio("Select Analysis Mode:", ["Single Image", "Batch (ZIP)"])
model_choice = col2.selectbox("Select Model Version:", list(MODEL_PATHS.keys()))

model = load_model(MODEL_PATHS[model_choice])

st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload image or ZIP file for inference", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file:
    if mode == "Single Image" and uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        image = Image.open(uploaded_file).convert("RGB")
        tensor = preprocess_image(image)
        results = predict(tensor, model)

        st.markdown("### 🧪 Prediction Results")
        for label, score in results.items():
            st.write(f"**{label}**: {score}%")

        st.markdown("### 💡 Suggestions")
        for suggestion in generate_suggestions(results):
            st.markdown(suggestion)

    elif mode == "Batch (ZIP)" and uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file) as archive:
            image_files = [f for f in archive.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                st.warning("No valid image files found in the ZIP.")
            for file_name in image_files:
                with archive.open(file_name) as file:
                    image = Image.open(file).convert("RGB")
                    st.markdown(f"#### 📂 File: `{file_name}`")
                    st.image(image, width=400)
                    tensor = preprocess_image(image)
                    results = predict(tensor, model)

                    st.markdown("**Prediction Scores:**")
                    for label, score in results.items():
                        st.write(f"{label}: {score}%")

                    st.markdown("**Suggestions:**")
                    for suggestion in generate_suggestions(results):
                        st.markdown(suggestion)
                    st.markdown("---")
    else:
        st.error("Invalid file type for selected mode.")
