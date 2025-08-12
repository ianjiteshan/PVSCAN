"""
Panel Analysis Model
Handles the AI model loading and prediction logic for solar panel inspection
"""
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanelAnalysisModel:
    """
    Handles loading and running inference with the solar panel inspection models
    """
    
    # Class configuration for scoring and suggestions
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

    # In the __init__ method
def __init__(self):
    """Initialize the panel analysis model"""
    self.script_dir = os.path.dirname(os.path.abspath(__file__))
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.models = {}
    
    # Ensure this transform definition is here
    self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    self._load_models()
    
    
    # def __init__(self, model_dir: str):
    #     """
    #     Initialize the panel analysis model
        
    #     Args:
    #         model_dir: Directory containing the model files
    #     """
    #     self.model_dir = model_dir
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.models = {}
    #     self.transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
        
    #     # Load models
    #     self._load_models()
    def _load_models(self):
        """Load the PyTorch models from the script's directory."""
        try:
            # Load v2.0 model
            v20_path = os.path.join(self.script_dir, "pvscan_mobilenetv3_v2.0.pth")
            print(f"DEBUG: Looking for model in: {v20_path}")

            if os.path.exists(v20_path):
                self.models['v2.0'] = self._load_single_model(v20_path)
                logger.info("Loaded model v2.0")
            else:
                logger.warning("Model v2.0 not found.")

            # Load v1.1 model
            v11_path = os.path.join(self.script_dir, "pvscan_mobilenetv3_v1.1.pth")
            if os.path.exists(v11_path):
                self.models['v1.1'] = self._load_single_model(v11_path)
                logger.info("Loaded model v1.1")
            else:
                logger.warning("Model v1.1 not found.")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    # def _load_models(self):
    #     """Load the PyTorch models"""
    #     try:
    #         # Load v2.0 model
    #         import os
    #         print("DEBUG: model_dir is:", self.model_dir)

    #         v20_path = os.path.join(self.model_dir, "pvscan_mobilenetv3_v2.0.pth")
    #         print("DEBUG: Looking for model in:", v20_path)
    #         print("DEBUG: Exists?", os.path.exists(v20_path))
    #         if os.path.exists(v20_path):
    #             self.models['v2.0'] = self._load_single_model(v20_path)
    #             logger.info("Loaded model v2.0")
            
    #         # Load v1.1 model
    #         v11_path = os.path.join(self.model_dir, "pvscan_mobilenetv3_v1.1.pth")
    #         if os.path.exists(v11_path):
    #             self.models['v1.1'] = self._load_single_model(v11_path)
    #             logger.info("Loaded model v1.1")
                
    #     except Exception as e:
    #         logger.error(f"Error loading models: {e}")
    #         raise
    
    def _load_single_model(self, model_path: str):
        """Load a single PyTorch model"""
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 8)  # 8 classes
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        return self.transform(image).unsqueeze(0)
    
    def predict(self, image_tensor: torch.Tensor, model_version: str = 'v2.0') -> Dict[str, float]:
        """Run prediction on preprocessed image tensor"""
        if model_version not in self.models:
            raise ValueError(f"Model version {model_version} not available")
        
        model = self.models[model_version]
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        outputs = outputs.squeeze().cpu().numpy()
        scores = [round(100 * (1 / (1 + np.exp(-x))), 1) for x in outputs]
        
        return {label: score for label, score in zip(self.CLASS_CONFIG.keys(), scores)}
    
    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Complete analysis of a solar panel image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing predictions, total score, and suggestions
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Get predictions from both models
            predictions_v20 = self.predict(image_tensor, 'v2.0')
            predictions_v11 = self.predict(image_tensor, 'v1.1')
            
            # Check panel detection threshold
            panel_detected_score = predictions_v20.get("Panel Detected", 0)
            if panel_detected_score < 50:
                return {
                    "success": False,
                    "error": f"Panel detection score ({panel_detected_score:.1f}%) below threshold (50%)",
                    "panel_detected": False
                }
            
            # Ensemble predictions
            final_predictions = self._ensemble_predictions(predictions_v11, predictions_v20)
            
            # Calculate total score
            total_score = self._calculate_total_score(final_predictions)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(final_predictions)
            
            return {
                "success": True,
                "panel_detected": True,
                "predictions": final_predictions,
                "total_score": round(total_score, 1),
                "condition": self._get_condition_label(total_score),
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "success": False,
                "error": str(e),
                "panel_detected": False
            }
    
    def _ensemble_predictions(self, pred_v11: Dict[str, float], pred_v20: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions from both models using weighted ensemble"""
        weights = {
            "Clean Panel": (0.3, 0.7), 
            "Physical Damage": (0.5, 0.5),
            "Electrical Damage": (0.5, 0.5), 
            "Snow Covered": (0.5, 0.5),
            "Water Obstruction": (0.2, 0.8), 
            "Foreign Particle Contamination": (0.5, 0.5),
            "Bird Interference": (0.3, 0.7), 
            "Panel Detected": (0.0, 1.0)
        }
        
        final_predictions = {}
        for label in self.CLASS_CONFIG.keys():
            if label in weights:
                w_v11, w_v20 = weights[label]
                score_v11 = pred_v11.get(label, 0)
                score_v20 = pred_v20.get(label, 0)
                final_predictions[label] = (w_v11 * score_v11) + (w_v20 * score_v20)
            else:
                final_predictions[label] = pred_v20.get(label, 0)
        
        return final_predictions
    
    def _calculate_total_score(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted total score based on predictions"""
        base_score = 100.0
        score_modifier = 0.0
        panel_detected_score = predictions.get("Panel Detected", 0)
        
        if panel_detected_score < 50:
            return 0.0
        
        for label, value in predictions.items():
            if label == "Panel Detected":
                continue
            if label in self.CLASS_CONFIG:
                for (low, high, ratio) in self.CLASS_CONFIG[label]["ranges"]:
                    if low <= value < high:
                        score_modifier += value * ratio
                        break
        
        total_score = base_score + score_modifier
        return max(0.0, min(total_score, 100.0))
    
    def _get_condition_label(self, total_score: float) -> str:
        """Get condition label based on total score"""
        if total_score >= 90:
            return "EXCELLENT"
        elif total_score >= 80:
            return "GOOD"
        elif total_score >= 70:
            return "AVERAGE"
        elif total_score >= 60:
            return "POOR"
        elif total_score >= 40:
            return "CRITICAL"
        else:
            return "CRITICAL"
    
    def _generate_suggestions(self, predictions: Dict[str, float]) -> List[str]:
        """Generate maintenance suggestions based on predictions"""
        suggestions = []
        
        clean_panel = predictions.get("Clean Panel", 0)
        physical_damage = predictions.get("Physical Damage", 0)
        electrical_damage = predictions.get("Electrical Damage", 0)
        
        # Clean panel check
        if clean_panel > 90 and physical_damage < 10 and electrical_damage < 10:
            return ["No cleaning required. Panel is in excellent condition."]
        
        if clean_panel < 70:
            suggestions.append(f"Cleaning required (Score: {clean_panel:.1f}%). Dirt accumulation may impact efficiency.")
        
        # Physical Damage
        if physical_damage > 70:
            suggestions.append(f"Critical physical damage ({physical_damage:.1f}%)! Immediate repair required.")
        elif physical_damage > 30:
            suggestions.append(f"High physical damage ({physical_damage:.1f}%). Repair strongly recommended.")
        elif physical_damage > 10:
            suggestions.append(f"Moderate physical damage ({physical_damage:.1f}%). Schedule maintenance soon.")
        
        # Electrical Damage
        if electrical_damage > 80:
            suggestions.append(f"Critical electrical damage ({electrical_damage:.1f}%)! Immediate expert consultation required.")
        elif electrical_damage > 50:
            suggestions.append(f"Severe electrical issue ({electrical_damage:.1f}%). Urgent inspection required.")
        elif electrical_damage > 30:
            suggestions.append(f"High electrical damage ({electrical_damage:.1f}%). Troubleshooting required soon.")
        
        # Other conditions
        snow = predictions.get("Snow Covered", 0)
        if snow > 50:
            suggestions.append(f"Panel covered with snow ({snow:.1f}%)! Removal needed.")
        
        water = predictions.get("Water Obstruction", 0)
        if water > 60:
            suggestions.append(f"Heavy water accumulation ({water:.1f}%). Cleaning recommended urgently.")
        
        contamination = predictions.get("Foreign Particle Contamination", 0)
        if contamination > 60:
            suggestions.append(f"Heavy foreign particle accumulation ({contamination:.1f}%). Clean the panel soon.")
        
        birds = predictions.get("Bird Interference", 0)
        if birds > 70:
            suggestions.append(f"Severe bird interference ({birds:.1f}%)! Install deterrents immediately.")
        
        return suggestions or ["No major issues detected. Panel appears to be in reasonable condition."]

