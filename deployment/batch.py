import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch.nn as nn
import torchvision.models as models
import numpy as np
import requests
import pandas as pd
import os
import zipfile
import tempfile
import shutil
from typing import List, Tuple, Optional, Union, Any # Added Any for broader type hint if needed

# --- Model Loading & Preprocessing ---

# Load Model Function (Messages moved to sidebar)
def load_model(model_url, device, num_classes=8):
    """
    Downloads a model state dictionary from a URL and loads it into a MobileNetV3-Large architecture.
    Status messages are shown in the sidebar.

    Args:
        model_url (str): The URL to the .pth model file.
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').
        num_classes (int): The number of output classes for the model's classifier.

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Raises:
        requests.exceptions.RequestException: If downloading the model fails.
        Exception: For other model loading errors.
    """
    try:
        st.sidebar.info(f"Downloading model: {os.path.basename(model_url)}...")
        response = requests.get(model_url)
        response.raise_for_status()
        st.sidebar.info("...download complete.")

        model_bytes = BytesIO(response.content)
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        st.sidebar.info("Loading model state...")
        state_dict = torch.load(model_bytes, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        st.sidebar.info("...model loaded successfully.")

        model.to(device)
        model.eval()
        return model
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error downloading model {os.path.basename(model_url)}: {e}")
        raise
    except Exception as e:
        st.sidebar.error(f"Error loading model {os.path.basename(model_url)}: {e}")
        raise

# Image Preprocessing Function
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses a PIL Image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Open Image Function (Handles both uploaded file object and file path)
# Corrected the type hint for the uploaded file object
def open_image(image_source: Union[str, BytesIO, Any]) -> Optional[Image.Image]:
    """
    Safely opens an image from a file path or an uploaded file object.

    Args:
        image_source: Path to the image file (str) or an uploaded file object (BytesIO or similar).

    Returns:
        PIL.Image.Image or None: The opened image in RGB format, or None if an error occurs.
    """
    image_name = "Uploaded File"
    # Try to get the name attribute if it exists (for uploaded files)
    if isinstance(image_source, str):
        image_name = os.path.basename(image_source)
    elif hasattr(image_source, 'name'):
        image_name = image_source.name

    try:
        image = Image.open(image_source).convert("RGB")
        return image
    except UnidentifiedImageError:
        st.warning(f"Cannot identify image file '{image_name}'. It might be corrupted or not a supported format. Skipping.")
        return None
    except Exception as e:
        st.warning(f"Could not open image file '{image_name}': {e}. Skipping.")
        return None

# Function to display a smaller version of the image
def display_compressed_image(image: Image.Image, max_width: int = 400):
    """Displays an image in Streamlit, resizing it if it exceeds a maximum width."""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image = image.resize((max_width, new_height))
    st.image(image, caption="Uploaded Image", use_container_width=False)

# Predict Function
def predict(image_tensor: torch.Tensor, model: nn.Module, device: torch.device) -> dict:
    """Performs inference on a preprocessed image tensor."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    outputs = outputs.squeeze().cpu().numpy()
    scores = [round(100 * (1 / (1 + np.exp(-x))), 1) for x in outputs]
    if len(scores) != len(CLASS_CONFIG.keys()):
         st.error(f"Model output size mismatch. Expected {len(CLASS_CONFIG.keys())}, got {len(scores)}.")
         return {}
    return {label: score for label, score in zip(CLASS_CONFIG.keys(), scores)}

# --- Scoring and Suggestions ---

# Scoring Configuration (Using the simplified one from the previous step)
CLASS_CONFIG = {
    "Panel Detected": {
        "ranges": [(0, 49, 0), (49, 60, 1.5), (60, 80, 1.2), (80, 90, 1.1), (90, 95, 1.05), (95, 100, 1)]
    },
    "Clean Panel": {
        "ranges": [(0, 2, -0.917), (2, 4, -0.925), (4, 6, -0.936), (6, 8, -0.948), (8, 10, -0.95),
                   (10, 12, -0.114), (12, 14, -0.126), (14, 16, -0.135), (16, 18, -0.147), (18, 20, -0.158),
                   (20, 22, -0.113), (22, 24, -0.125), (24, 26, -0.137), (26, 28, -0.148), (28, 30, -0.159),
                   (30, 32, -0.114), (32, 34, -0.127), (34, 36, -0.135), (36, 38, -0.142), (38, 40, -0.158),
                   (40, 42, -0.217), (42, 44, -0.128), (44, 46, -0.139), (46, 48, -0.146), (48, 50, -0.153),
                   (50, 52, -0.113), (52, 54, -0.127), (54, 56, -0.138), (56, 58, -0.142), (58, 60, -0.159),
                   (60, 62, -0.116), (62, 64, -0.128), (64, 66, -0.137), (66, 68, -0.145), (68, 70, -0.156),
                   (70, 72, -0.057), (72, 74, -0.063), (74, 76, -0.071), (76, 78, -0.082), (78, 80, -0.094),
                   (80, 82, 0.014), (82, 84, 0.023), (84, 86, 0.031), (86, 88, 0.042), (88, 90, 0.052),
                   (90, 92, 0.017), (92, 94, 0.016), (94, 96, 0.018), (96, 98, 0.017), (98, 100, 0.019), (100, 101, 0.018)]
    },
    "Physical Damage": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.817), (4, 6, -0.823), (6, 8, -0.834), (8, 10, -0.845),
                   (10, 12, -0.716), (12, 14, -0.724), (14, 16, -0.735), (16, 18, -0.746), (18, 20, -0.753),
                   (20, 22, -0.517), (22, 24, -0.525), (24, 26, -0.536), (26, 28, -0.548), (28, 30, -0.559),
                   (30, 32, -0.214), (32, 34, -0.226), (34, 36, -0.237), (36, 38, -0.243), (38, 40, -0.252),
                   (40, 42, -0.217), (42, 44, -0.226), (44, 46, -0.232), (46, 48, -0.245), (48, 50, -0.258),
                   (50, 52, -0.213), (52, 54, -0.221), (54, 56, -0.232), (56, 58, -0.243), (58, 60, -0.256),
                   (60, 62, -0.262), (62, 64, -0.273), (64, 66, -0.284), (66, 68, -0.292), (68, 70, -0.259),
                   (70, 72, -0.263), (72, 74, -0.271), (74, 76, -0.282), (76, 78, -0.293), (78, 80, -0.257),
                   (80, 82, -0.262), (82, 84, -0.273), (84, 86, -0.284), (86, 88, -0.291), (88, 90, -0.259),
                   (90, 92, -0.261), (92, 94, -0.272), (94, 96, -0.283), (96, 98, -0.256), (98, 100, -0.247), (100, 101, -0.353)]
    },
    "Electrical Damage": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.812), (4, 6, -0.823), (6, 8, -0.831), (8, 10, -0.842),
                   (10, 12, -0.713), (12, 14, -0.724), (14, 16, -0.736), (16, 18, -0.747), (18, 20, -0.758),
                   (20, 22, -0.312), (22, 24, -0.323), (24, 26, -0.534), (26, 28, -0.346), (28, 30, -0.357),
                   (30, 32, -0.217), (32, 34, -0.228), (34, 36, -0.539), (36, 38, -0.341), (38, 40, -0.352),
                   (40, 42, -0.213), (42, 44, -0.224), (44, 46, -0.335), (46, 48, -0.346), (48, 50, -0.257),
                   (50, 52, -0.214), (52, 54, -0.326), (54, 56, -0.237), (56, 58, -0.238), (58, 60, -0.259),
                   (60, 62, -0.215), (62, 64, -0.327), (64, 66, -0.238), (66, 68, -0.249), (68, 70, -0.251),
                   (70, 72, -0.315), (72, 74, -0.327), (74, 76, -0.238), (76, 78, -0.249), (78, 80, -0.251),
                   (80, 82, -0.313), (82, 84, -0.325), (84, 86, -0.236), (86, 88, -0.247), (88, 90, -0.258),
                   (90, 92, -0.363), (92, 94, -0.372), (94, 96, -0.358), (96, 98, -0.349), (98, 100, -0.351), (100, 101, -0.352)]
    },
    "Snow Covered": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.813), (4, 6, -0.824), (6, 8, -0.836), (8, 10, -0.847),
                   (10, 12, -0.719), (12, 14, -0.721), (14, 16, -0.732), (16, 18, -0.743), (18, 20, -0.754),
                   (20, 22, -0.314), (22, 24, -0.325), (24, 26, -0.336), (26, 28, -0.347), (28, 30, -0.358),
                   (30, 32, -0.317), (32, 34, -0.329), (34, 36, -0.331), (36, 38, -0.342), (38, 40, -0.354),
                   (40, 42, -0.316), (42, 44, -0.327), (44, 46, -0.338), (46, 48, -0.349), (48, 50, -0.351),
                   (50, 52, -0.318), (52, 54, -0.329), (54, 56, -0.331), (56, 58, -0.342), (58, 60, -0.353),
                   (60, 62, -0.319), (62, 64, -0.321), (64, 66, -0.332), (66, 68, -0.344), (68, 70, -0.355),
                   (70, 72, -0.319), (72, 74, -0.321), (74, 76, -0.332), (76, 78, -0.343), (78, 80, -0.354),
                   (80, 82, -0.319), (82, 84, -0.328), (84, 86, -0.339), (86, 88, -0.341), (88, 90, -0.352),
                   (90, 92, -0.362), (92, 94, -0.373), (94, 96, -0.384), (96, 98, -0.356), (98, 100, -0.349), (100, 101, -0.558)]
    },
    "Water Obstruction": {
        "ranges": [(0, 2, -0.95), (2, 4, -0.817), (4, 6, -0.828), (6, 8, -0.839), (8, 10, -0.841),
                   (10, 12, -0.516), (12, 14, -0.527), (14, 16, -0.538), (16, 18, -0.549), (18, 20, -0.551),
                   (20, 22, -0.215), (22, 24, -0.226), (24, 26, -0.237), (26, 28, -0.248), (28, 30, -0.259),
                   (30, 32, -0.216), (32, 34, -0.227), (34, 36, -0.238), (36, 38, -0.249), (38, 40, -0.251),
                   (40, 42, -0.217), (42, 44, -0.228), (44, 46, -0.239), (46, 48, -0.241), (48, 50, -0.252),
                   (50, 52, -0.218), (52, 54, -0.229), (54, 56, -0.231), (56, 58, -0.242), (58, 60, -0.253),
                   (60, 62, -0.219), (62, 64, -0.221), (64, 66, -0.232), (66, 68, -0.243), (68, 70, -0.254),
                   (70, 72, -0.217), (72, 74, -0.228), (74, 76, -0.239), (76, 78, -0.241), (78, 80, -0.252),
                   (80, 82, -0.218), (82, 84, -0.229), (84, 86, -0.231), (86, 88, -0.242), (88, 90, -0.253),
                   (90, 92, -0.262), (92, 94, -0.273), (94, 96, -0.264), (96, 98, -0.255), (98, 100, -0.346), (100, 101, -0.354)]
    },
    "Foreign Particle Contamination": {
        "ranges": [(0, 2, 0.013), (2, 4, -0.517), (4, 6, -0.528), (6, 8, -0.539), (8, 10, -0.541),
                   (10, 12, -0.215), (12, 14, -0.226), (14, 16, -0.237), (16, 18, -0.248), (18, 20, -0.259),
                   (20, 22, -0.216), (22, 24, -0.227), (24, 26, -0.218), (26, 28, -0.229), (28, 30, -0.231),
                   (30, 32, -0.217), (32, 34, -0.228), (34, 36, -0.239), (36, 38, -0.241), (38, 40, -0.252),
                   (40, 42, -0.218), (42, 44, -0.229), (44, 46, -0.217), (46, 48, -0.228), (48, 50, -0.239),
                   (50, 52, -0.219), (52, 54, -0.221), (54, 56, -0.219), (56, 58, -0.221), (58, 60, -0.232),
                   (60, 62, -0.111), (62, 64, -0.122), (64, 66, -0.133), (66, 68, -0.144), (68, 70, -0.155),
                   (70, 72, -0.112), (72, 74, -0.123), (74, 76, -0.134), (76, 78, -0.145), (78, 80, -0.156),
                   (80, 82, -0.113), (82, 84, -0.124), (84, 86, -0.135), (86, 88, -0.146), (88, 90, -0.157),
                   (90, 92, -0.114), (92, 94, -0.125), (94, 96, -0.136), (96, 98, -0.147), (98, 100, -0.358), (100, 101, -0.369)]
    },
    "Bird Interference": {
        "ranges": [(0, 2, 0.017), (2, 4, -0.516), (4, 6, -0.527), (6, 8, -0.538), (8, 10, -0.549),
                   (10, 12, -0.319), (12, 14, -0.321), (14, 16, -0.332), (16, 18, -0.343), (18, 20, -0.354),
                   (20, 22, -0.213), (22, 24, -0.224), (24, 26, -0.216), (26, 28, -0.227), (28, 30, -0.238),
                   (30, 32, -0.214), (32, 34, -0.225), (34, 36, -0.236), (36, 38, -0.247), (38, 40, -0.258),
                   (40, 42, -0.215), (42, 44, -0.226), (44, 46, -0.237), (46, 48, -0.248), (48, 50, -0.259),
                   (50, 52, -0.216), (52, 54, -0.227), (54, 56, -0.238), (56, 58, -0.249), (58, 60, -0.251),
                   (60, 62, -0.217), (62, 64, -0.228), (64, 66, -0.239), (66, 68, -0.241), (68, 70, -0.252),
                   (70, 72, -0.218), (72, 74, -0.229), (74, 76, -0.231), (76, 78, -0.242), (78, 80, -0.253),
                   (80, 82, -0.219), (82, 84, -0.221), (84, 86, -0.232), (86, 88, -0.243), (88, 90, -0.254),
                   (90, 92, -0.211), (92, 94, -0.222), (94, 96, -0.233), (96, 98, -0.244), (98, 100, -0.255), (100, 101, -0.266)]
    }
}


# Function to calculate the total score
def calculate_total_score(predictions: dict) -> float:
    """Calculates a weighted total score based on individual class predictions."""
    base_score = 100.0
    score_modifier = 0.0
    panel_detected_score = predictions.get("Panel Detected", 0)
    if panel_detected_score < 50: return 0.0 # Panel not detected well

    for label, value in predictions.items():
        if label == "Panel Detected": continue
        if label in CLASS_CONFIG:
            for (low, high, ratio) in CLASS_CONFIG[label]["ranges"]:
                # Ensure comparison works correctly at boundaries (use < high)
                if low <= value < high:
                    score_modifier += value * ratio
                    break
    total_score = base_score + score_modifier
    return max(0.0, min(total_score, 100.0)) # Clamp score

# Display Total Score and Condition Classification
def display_total_score(total_score: float):
    """Displays the total score and a qualitative condition assessment."""
    st.markdown(f"### Total Score: **{total_score:.1f}/100**")
    if total_score >= 90: st.success("✅ EXCELLENT CONDITION")
    elif total_score >= 80: st.success("👍 GOOD CONDITION")
    elif total_score >= 70: st.warning("👌 AVERAGE CONDITION")
    elif total_score >= 50: st.error("⚠️ POOR CONDITION")
    else: st.error("🚨 CRITICAL CONDITION")

# Cleaning Suggestions Based on Scores (Using the detailed version from user example)
def cleaning_suggestions(predictions: dict) -> List[str]:
    """Generates maintenance/cleaning suggestions based on prediction scores."""
    suggestions = []
    severity_order = {"🔴": 1, "🟠": 2, "🟡": 3, "🟢": 4} # For sorting

    # Clean panel check
    clean_panel = predictions.get("Clean Panel", 0)
    physical_damage = predictions.get("Physical Damage", 0)
    electrical_damage = predictions.get("Electrical Damage", 0)

    if clean_panel > 90 and physical_damage < 10 and electrical_damage < 10:
        return ["🟢 No cleaning required. Panel is in excellent condition."]
    if clean_panel < 70:
        suggestions.append(f"🟠 Cleaning required (Score: {clean_panel:.1f}%). Dirt accumulation may impact efficiency.")

    # Physical Damage
    damage = physical_damage
    if damage > 70: suggestions.append(f"🔴 Critical physical damage ({damage:.1f}%)! Immediate repair required.")
    elif damage > 30: suggestions.append(f"🟠 High physical damage ({damage:.1f}%). Repair strongly recommended.")
    elif damage > 10: suggestions.append(f"🟡 Moderate physical damage ({damage:.1f}%). Schedule maintenance soon.")
    elif damage > 5: suggestions.append(f"🟡 Noticeable physical damage ({damage:.1f}%). Preventive action advised.")

    # Electrical Damage
    electrical = electrical_damage
    if electrical > 80: suggestions.append(f"🔴 Critical electrical damage ({electrical:.1f}%)! Immediate expert consultation required.")
    elif electrical > 50: suggestions.append(f"🔴 Severe electrical issue ({electrical:.1f}%). Urgent inspection required.")
    elif electrical > 30: suggestions.append(f"🟠 High electrical damage ({electrical:.1f}%). Troubleshooting required soon.")
    elif electrical > 10: suggestions.append(f"🟠 Low electrical concern detected ({electrical:.1f}%). Monitor for worsening symptoms.")

    # Snow Coverage
    snow = predictions.get("Snow Covered", 0)
    if snow > 50: suggestions.append(f"🔴 Panel fully covered with snow ({snow:.1f}%)! Immediate removal needed.")
    elif snow > 20: suggestions.append(f"🟠 Significant snow presence ({snow:.1f}%). Consider clearing.")
    elif snow > 10: suggestions.append(f"🟡 Low snow presence ({snow:.1f}%). Monitor.")

    # Water Obstruction
    water = predictions.get("Water Obstruction", 0)
    if water > 60: suggestions.append(f"🔴 Heavy water accumulation ({water:.1f}%). Cleaning recommended urgently.")
    elif water > 30: suggestions.append(f"🟠 Moderate water presence ({water:.1f}%). Consider clearing soon.")
    elif water > 10: suggestions.append(f"🟡 Light water presence ({water:.1f}%). Monitor.")

    # Foreign Particle Contamination
    contamination = predictions.get("Foreign Particle Contamination", 0)
    if contamination > 60: suggestions.append(f"🔴 Heavy foreign particle accumulation ({contamination:.1f}%). Clean the panel soon.")
    elif contamination > 30: suggestions.append(f"🟠 Moderate foreign particle accumulation detected ({contamination:.1f}%). Cleaning recommended.")
    elif contamination > 10: suggestions.append(f"🟢 Light foreign particle contamination present ({contamination:.1f}%). Preventive cleaning advised.")

    # Bird Interference
    birds = predictions.get("Bird Interference", 0)
    if birds > 70: suggestions.append(f"🔴 Severe bird interference ({birds:.1f}%)! Install deterrents immediately.")
    elif birds > 30: suggestions.append(f"🟠 Moderate bird interference ({birds:.1f}%). Deterrents may be needed.")
    elif birds > 10: suggestions.append(f"🟡 Light bird activity ({birds:.1f}%). Monitor and take action if needed.")

    if not suggestions:
        return ["🟢 No major issues detected. Panel appears to be in reasonable condition."]

    # Sort suggestions by severity
    sorted_suggestions = sorted(suggestions, key=lambda s: severity_order.get(s.split(" ")[0], 5))
    return sorted_suggestions


# --- Image Processing Logic ---

# Function to process a single image and return predictions
def process_single_image(image: Image.Image,
                         panel_detection_model: nn.Module,
                         inspection_model_v11: nn.Module,
                         inspection_model_v20: nn.Module,
                         device: torch.device) -> dict:
    """Processes a single image through detection and inspection models."""
    image_tensor = preprocess_image(image)
    predictions_v20 = predict(image_tensor, inspection_model_v20, device)
    if not predictions_v20: raise ValueError("Model prediction failed for v2.0.")

    panel_detected_score = predictions_v20.get("Panel Detected", 0)
    panel_detection_threshold = 50
    if panel_detected_score < panel_detection_threshold:
        raise ValueError(f"Panel detection score ({panel_detected_score:.1f}%) below threshold ({panel_detection_threshold}%)")

    predictions_v11 = predict(image_tensor, inspection_model_v11, device)
    if not predictions_v11: raise ValueError("Model prediction failed for v1.1.")

    # Ensemble/Combine Predictions
    final_predictions = {}
    weights = {
        "Clean Panel": (0.3, 0.7), "Physical Damage": (0.5, 0.5),
        "Electrical Damage": (0.5, 0.5), "Snow Covered": (0.5, 0.5),
        "Water Obstruction": (0.2, 0.8), "Foreign Particle Contamination": (0.5, 0.5),
        "Bird Interference": (0.3, 0.7), "Panel Detected": (0.0, 1.0)
    }
    for label in CLASS_CONFIG.keys():
        if label in weights:
            w_v11, w_v20 = weights[label]
            score_v11 = predictions_v11.get(label, 0)
            score_v20 = predictions_v20.get(label, 0)
            final_predictions[label] = (w_v11 * score_v11) + (w_v20 * score_v20)
        else:
            final_predictions[label] = predictions_v20.get(label, 0)
    return final_predictions


# Function to process a batch of images from file paths
def process_batch_images(image_files: List[str],
                         panel_detection_model: nn.Module,
                         inspection_model_v11: nn.Module,
                         inspection_model_v20: nn.Module,
                         device: torch.device,
                         progress_bar) -> Tuple[List[dict], List[str]]:
    """Processes a list of image file paths in batch."""
    all_results = []
    error_messages = []
    total_files = len(image_files)

    for i, image_file_path in enumerate(image_files):
        image_name = os.path.basename(image_file_path)
        try:
            progress_text = f"Processing image {i+1}/{total_files}: {image_name}"
            progress_bar.progress((i + 1) / total_files, text=progress_text)

            image = open_image(image_file_path) # Use path here
            if image is None:
                error_messages.append(f"'{image_name}': Failed to open or invalid image file.")
                continue

            predictions = process_single_image(
                image, panel_detection_model, inspection_model_v11, inspection_model_v20, device
            )
            total_score = calculate_total_score(predictions)
            result_data = {
                "Image Name": image_name,
                "Total Score": round(total_score, 2),
                **{label: round(score, 2) for label, score in predictions.items()}
            }
            all_results.append(result_data)

        except ValueError as ve:
             st.warning(f"Skipping '{image_name}': {ve}")
             error_messages.append(f"'{image_name}': Skipped ({ve})")
        except Exception as e:
            st.error(f"Unexpected error processing image '{image_name}': {e}")
            error_messages.append(f"'{image_name}': Unexpected Error ({e})")

    return all_results, error_messages

# --- Streamlit UI ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="SPICE.AI", layout="wide")

    # --- Device Configuration (Moved from Sidebar to hidden) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Logo Display (Responsive) ---
    st.markdown("""
        <style>
        .desktop-logo { display: block; margin-bottom: 20px; }
        .mobile-logo { display: none; margin-bottom: 20px; }
        @media (max-width: 768px) { .desktop-logo { display: none; } .mobile-logo { display: block; } }
        .desktop-logo img, .mobile-logo img { display: block; margin-left: auto; margin-right: auto; max-width: 100%; height: auto; }
        div[data-testid="stDataFrame"] th { text-align: center !important; }
        div[data-testid="stDataFrame"] td { text-align: center !important; }
        </style>
    """, unsafe_allow_html=True)
    desktop_logo_url = "https://github.com/ianjiteshan/PVSCAN/blob/main/deployment/logo_comp.png?raw=true"
    mobile_logo_url = "https://github.com/ianjiteshan/PVSCAN/blob/main/deployment/logo_phone.png?raw=true"
    st.markdown(f"""
        <div class="desktop-logo"><img src="{desktop_logo_url}" alt="PVSCAN Desktop Logo"></div>
        <div class="mobile-logo"><img src="{mobile_logo_url}" alt="PVSCAN Mobile Logo"></div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # --- Model Loading (Cached, Status messages moved to main area) ---
    @st.cache_resource
    def load_all_models(device):
        # Loads all required models and caches them.
        models_dict = {}
        try:
            model_url_v20 = "https://raw.githubusercontent.com/ianjiteshan/PVSCAN/main/deployment/pvscan_mobilenetv3_v2.0.pth"
            models_dict['inspection_model_v20'] = load_model(model_url_v20, device, num_classes=8)
            # Reuse v2.0 model for panel detection as well
            models_dict['panel_detection_model'] = models_dict['inspection_model_v20']
        except Exception:
            # Error logged in load_model
            models_dict['inspection_model_v20'] = None
            models_dict['panel_detection_model'] = None
        try:
            model_url_v11 = "https://raw.githubusercontent.com/ianjiteshan/PVSCAN/main/deployment/pvscan_mobilenetv3_v1.1.pth"
            models_dict['inspection_model_v11'] = load_model(model_url_v11, device, num_classes=8)
        except Exception:
            # Error logged in load_model
            models_dict['inspection_model_v11'] = None
        return models_dict

    # Show model loading message in main area
    with st.spinner("Loading models..."):
        models_loaded = load_all_models(device)
    
    # Check if essential models loaded successfully
    if not models_loaded.get('panel_detection_model') or \
       not models_loaded.get('inspection_model_v11') or \
       not models_loaded.get('inspection_model_v20'):
        st.error("🔴 One or more essential models failed to load. Application cannot continue.")
        st.stop() # Stop execution if models aren't loaded
    else:
        st.success("✅ All models loaded successfully")

    # Assign models from the loaded dictionary
    panel_detection_model = models_loaded['panel_detection_model']
    inspection_model_v11 = models_loaded['inspection_model_v11']
    inspection_model_v20 = models_loaded['inspection_model_v20']

    # --- Input Mode Selection ---
    st.header("Select Input Mode")
    input_mode = st.radio(
        "Choose how to provide images:",
        ["Single Image Upload", "Zip File Batch Upload"],
        key="input_mode_radio",
        horizontal=True
    )
    st.markdown("---") # Separator

    # --- UI Logic for Selected Mode ---

    if input_mode == "Single Image Upload":
        st.subheader("Upload a Single Solar Panel Image")
        uploaded_file = st.file_uploader(
            "Select an image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="single_uploader" # Unique key for this uploader
        )

        # Placeholders for single image results
        image_placeholder = st.empty()
        results_placeholder = st.container() # Use container for multiple elements like table and score
        suggestions_placeholder = st.container() # Separate container for suggestions

        if uploaded_file is not None:
            # Clear previous results before processing new image
            # This ensures that old results disappear when a new file is uploaded
            results_placeholder.empty()
            suggestions_placeholder.empty()
            image_placeholder.empty()

            image = open_image(uploaded_file) # Use uploaded file object directly
            if image:
                # Display the uploaded image
                with image_placeholder.container(): # Use container to group subheader and image
                    st.subheader("Uploaded Image:")
                    # Display image, limiting width for consistency
                    st.image(image, caption=uploaded_file.name, width=400)

                try:
                    # Show spinner during analysis
                    with st.spinner("Analyzing image... Please wait."):
                        final_predictions = process_single_image(
                            image, panel_detection_model, inspection_model_v11, inspection_model_v20, device
                        )

                    # Display results in the results placeholder
                    with results_placeholder:
                        st.subheader("Inspection Analysis")
                        # Create and display DataFrame
                        df = pd.DataFrame.from_dict(final_predictions, orient='index', columns=['Score (%)'])
                        df['Score (%)'] = df['Score (%)'].apply(lambda x: f"{x:.1f}") # Format score
                        st.dataframe(df.style.set_properties(**{'text-align': 'center'}), use_container_width=True) # Center align

                        # Calculate and display total score
                        total_score = calculate_total_score(final_predictions)
                        display_total_score(total_score)

                    # Display suggestions in the suggestions placeholder
                    with suggestions_placeholder:
                         st.subheader("Suggestions")
                         suggestions = cleaning_suggestions(final_predictions)
                         # Display suggestions as a list
                         for suggestion in suggestions:
                             st.markdown(f"- {suggestion}")

                except ValueError as e: # Catch panel detection or prediction errors
                    results_placeholder.error(f"⚠️ Analysis Error: {e}")
                except Exception as e:
                    results_placeholder.error(f"🔴 An unexpected error occurred during analysis: {e}")
                    # Log error to main area instead of sidebar
                    st.error(f"Error details: {str(e)}")
            else:
                 # Error opening image is handled by open_image, show warning
                 image_placeholder.warning(f"Could not process the uploaded file: {uploaded_file.name}")
        else:
            # Show initial instruction message if no file is uploaded yet
            st.info("⬆️ Upload an image file above to start the analysis.")


    elif input_mode == "Zip File Batch Upload":
        st.subheader("Upload Zip Archive for Batch Processing")
        uploaded_zip = st.file_uploader(
            "Select a Zip file containing solar panel images (.png, .jpg, .jpeg, .webp)",
            type="zip",
            accept_multiple_files=False,
            key="zip_uploader" # Unique key for this uploader
        )

        # Placeholders for batch results - helps manage UI updates
        results_placeholder_batch = st.empty()
        download_placeholder_batch = st.empty()
        error_placeholder_batch = st.empty()

        if uploaded_zip is not None:
            # Clear previous batch results when a new zip is uploaded
            results_placeholder_batch.empty()
            download_placeholder_batch.empty()
            error_placeholder_batch.empty()

            st.info(f"Processing uploaded zip file: **{uploaded_zip.name}**")
            temp_dir = None # Initialize temporary directory variable
            try:
                # Extract zip file contents
                with st.spinner(f"Extracting images from {uploaded_zip.name}..."):
                    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                        # Create a unique temporary directory
                        temp_dir = tempfile.mkdtemp(prefix="spice_ai_zip_")
                        zip_ref.extractall(temp_dir)

                # Find valid image files within the extracted directory (recursive search)
                image_files = []
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                             image_files.append(os.path.join(root, f))

                # Check if any images were found
                if not image_files:
                    results_placeholder_batch.warning("⚠️ No valid image files (.png, .jpg, .jpeg, .webp) found in the zip archive.")
                else:
                    st.info(f"Found {len(image_files)} images. Starting analysis...")
                    # Create progress bar in the main area
                    progress_bar = st.progress(0, text="Starting analysis...")

                    # Process the batch of images
                    all_results, error_messages = process_batch_images(
                        image_files, panel_detection_model, inspection_model_v11, inspection_model_v20, device, progress_bar
                    )
                    progress_bar.empty() # Clear progress bar after completion

                    # Display results table if any images were processed successfully
                    if all_results:
                        with results_placeholder_batch.container(): # Group subheader and table
                            st.subheader("Batch Processing Results")
                            df_results = pd.DataFrame(all_results)
                            # Reorder columns for better readability
                            cols_order = ["Image Name", "Total Score"] + [k for k in CLASS_CONFIG.keys() if k != "Panel Detected"] + ["Panel Detected"]
                            df_results = df_results[[col for col in cols_order if col in df_results.columns]] # Ensure columns exist
                            st.dataframe(df_results, use_container_width=True)

                        # Provide CSV download button
                        csv_data = df_results.to_csv(index=False).encode('utf-8')
                        download_placeholder_batch.download_button(
                            label="⬇️ Download Results as CSV", data=csv_data,
                            file_name=f"spice_ai_{os.path.splitext(uploaded_zip.name)[0]}_results.csv",
                            mime='text/csv', key='download_csv_button_batch' # Unique key
                        )
                    else:
                         # Message if no images could be processed (e.g., all failed detection)
                         results_placeholder_batch.warning("⚠️ No images were successfully processed from the zip file.")

                    # Report errors encountered during processing
                    if error_messages:
                        with error_placeholder_batch.container(): # Group error header and messages
                            st.error("❗️ Errors occurred during processing:")
                            error_md = ""
                            for msg in error_messages: error_md += f"- {msg}\n"
                            st.markdown(error_md)

            except zipfile.BadZipFile:
                results_placeholder_batch.error("🔴 Invalid or corrupted zip file.")
            except Exception as e:
                results_placeholder_batch.error(f"🔴 An unexpected error occurred during zip processing: {e}")
                # Log error to main area instead of sidebar
                st.error(f"Error details: {str(e)}")
            finally:
                # --- Cleanup Temporary Directory ---
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        # Move cleanup error to main area
                        st.warning(f"Could not remove temporary directory: {e}")
        else:
            # Show initial instruction message if no zip file is uploaded yet
            st.info("⬆️ Upload a zip file above to start batch analysis.")


# Entry point for the script
if __name__ == "__main__":
    main()