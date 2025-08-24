"""
Analysis Routes
Handles API endpoints for solar panel image analysis
"""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import os
import zipfile
import tempfile
import shutil
from typing import List, Dict
import logging

from src.models.panel_analysis import PanelAnalysisModel
from src.utils import allowed_file


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__)

# Global model instance
model_instance = None

def get_model():
    """Get or create model instance"""
    global model_instance
    if model_instance is None:
        model_instance = PanelAnalysisModel()
    return model_instance



def safe_open_image(file_path: str) -> Image.Image:
    """Safely open an image file"""
    try:
        image = Image.open(file_path).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file. It might be corrupted or not a supported format.")
    except Exception as e:
        raise ValueError(f"Could not open image file: {e}")

@analysis_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return jsonify({
            "status": "healthy",
            "models_loaded": list(model.models.keys()),
            "device": str(model.device)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

import numpy as np

def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj

@analysis_bp.route("/analyze-single", methods=["POST"])
def analyze_single_image():
    """Analyze a single uploaded image"""
    try:
        # Check if file is present
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_path)
            
            # Open and analyze image
            image = safe_open_image(temp_path)
            model = get_model()
            result = model.analyze_image(image)
            
            # Add filename to result
            result["filename"] = filename
            
            # Convert numpy types to native Python types
            result = convert_to_native_types(result)
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analyze_single_image: {e}")
        return jsonify({"error": "Internal server error"}), 500

@analysis_bp.route('/analyze-batch', methods=['POST'])
def analyze_batch_images():
    """Analyze multiple images from a zip file"""
    try:
        # Check if file is present
        if 'zipfile' not in request.files:
            return jsonify({"error": "No zip file provided"}), 400
        
        file = request.files['zipfile']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.zip'):
            return jsonify({"error": "File must be a zip archive"}), 400
        
        # Save uploaded zip file temporarily
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'upload.zip')
        extract_dir = os.path.join(temp_dir, 'extracted')
        
        try:
            file.save(zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find all image files
            image_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                return jsonify({"error": "No valid image files found in zip"}), 400
            
            # Analyze each image
            model = get_model()
            results = []
            errors = []
            
            for i, image_path in enumerate(image_files):
                try:
                    filename = os.path.basename(image_path)
                    image = safe_open_image(image_path)
                    result = model.analyze_image(image)
                    
                    # Add metadata
                    result["filename"] = filename
                    result["image_number"] = i + 1
                    
                    # Convert numpy types to native Python types
                    result = convert_to_native_types(result)
                    
                    if result["success"]:
                        results.append(result)
                    else:
                        errors.append(f'{filename}: {result.get("error", "Unknown error")}')
                        
                except Exception as e:
                    errors.append(f"{os.path.basename(image_path)}: {str(e)}")
            
            # Calculate summary statistics
            if results:
                total_scores = [r["total_score"] for r in results]
                summary = {
                    "total_images": len(image_files),
                    "successful_analyses": len(results),
                    "failed_analyses": len(errors),
                    "average_score": round(sum(total_scores) / len(total_scores), 1),
                    "min_score": min(total_scores),
                    "max_score": max(total_scores)
                }
            else:
                summary = {
                    "total_images": len(image_files),
                    "successful_analyses": 0,
                    "failed_analyses": len(errors),
                    "average_score": 0,
                    "min_score": 0,
                    "max_score": 0
                }
            
            # Convert numpy types in summary to native Python types
            summary = convert_to_native_types(summary)

            return jsonify({
                "success": True,
                "summary": summary,
                "results": results,
                "errors": errors
            })
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Error in analyze_batch_images: {e}")
        return jsonify({"error": "Internal server error"}), 500

@analysis_bp.route("/model-info", methods=["GET"])
def get_model_info():
    """Get information about loaded models"""
    try:
        model = get_model()
        info = {
            "available_models": list(model.models.keys()),
            "device": str(model.device),
            "classes": list(model.CLASS_CONFIG.keys())
        }
        return jsonify(convert_to_native_types(info))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@analysis_bp.route("/efficiency-metrics", methods=["POST"])
def calculate_efficiency_metrics():
    """Calculate efficiency metrics from analysis results"""
    try:
        data = request.get_json()
        if not data or "results" not in data:
            return jsonify({"error": "No results data provided"}), 400
        
        results = data["results"]
        if not results:
            return jsonify({"error": "Empty results array"}), 400
        
        # Calculate various efficiency metrics
        total_panels = len(results)
        panel_scores = [r.get("total_score", 0) for r in results]
        
        # Score distribution
        excellent = sum(1 for score in panel_scores if score >= 90)
        good = sum(1 for score in panel_scores if 80 <= score < 90)
        average = sum(1 for score in panel_scores if 70 <= score < 80)
        poor = sum(1 for score in panel_scores if 60 <= score < 70)
        critical = sum(1 for score in panel_scores if score < 60)
        
        # Issue analysis
        issues = {
            "physical_damage": 0,
            "electrical_damage": 0,
            "snow_covered": 0,
            "water_obstruction": 0,
            "contamination": 0,
            "bird_interference": 0
        }
        
        for result in results:
            predictions = result.get("predictions", {})
            if predictions.get("Physical Damage", 0) > 30:
                issues["physical_damage"] += 1
            if predictions.get("Electrical Damage", 0) > 30:
                issues["electrical_damage"] += 1
            if predictions.get("Snow Covered", 0) > 30:
                issues["snow_covered"] += 1
            if predictions.get("Water Obstruction", 0) > 30:
                issues["water_obstruction"] += 1
            if predictions.get("Foreign Particle Contamination", 0) > 30:
                issues["contamination"] += 1
            if predictions.get("Bird Interference", 0) > 30:
                issues["bird_interference"] += 1
        
        metrics = {
            "total_panels": total_panels,
            "average_score": round(sum(panel_scores) / total_panels, 1),
            "median_score": round(sorted(panel_scores)[total_panels // 2], 1),
            "min_score": min(panel_scores),
            "max_score": max(panel_scores),
            "score_distribution": {
                "excellent": {"count": excellent, "percentage": round(excellent / total_panels * 100, 1)},
                "good": {"count": good, "percentage": round(good / total_panels * 100, 1)},
                "average": {"count": average, "percentage": round(average / total_panels * 100, 1)},
                "poor": {"count": poor, "percentage": round(poor / total_panels * 100, 1)},
                "critical": {"count": critical, "percentage": round(critical / total_panels * 100, 1)}
            },
            "common_issues": issues,
            "efficiency_rating": _calculate_efficiency_rating(sum(panel_scores) / total_panels),
            "maintenance_priority": _get_maintenance_priority(issues, total_panels)
        }
        
        return jsonify(convert_to_native_types(metrics))
        
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {e}")
        return jsonify({"error": "Internal server error"}), 500

def _calculate_efficiency_rating(average_score: float) -> str:
    """Calculate overall efficiency rating"""
    if average_score >= 85:
        return "OPTIMAL"
    elif average_score >= 75:
        return "GOOD"
    elif average_score >= 65:
        return "MODERATE"
    elif average_score >= 50:
        return "POOR"
    else:
        return "CRITICAL"

def _get_maintenance_priority(issues: Dict[str, int], total_panels: int) -> List[Dict]:
    """Get maintenance priorities based on issue frequency"""
    priorities = []
    
    for issue_type, count in issues.items():
        if count > 0:
            percentage = round(count / total_panels * 100, 1)
            priority = "HIGH" if percentage > 20 else "MEDIUM" if percentage > 10 else "LOW"
            priorities.append({
                "issue": issue_type.replace('_', ' ').title(),
                "affected_panels": count,
                "percentage": percentage,
                "priority": priority
            })
    
    # Sort by percentage descending
    priorities.sort(key=lambda x: x['percentage'], reverse=True)
    return priorities

