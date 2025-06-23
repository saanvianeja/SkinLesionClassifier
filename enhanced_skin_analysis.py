"""
Enhanced Skin Lesion Analysis combining CNN predictions with metadata analysis
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import os
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSkinAnalysis:
    def __init__(self, model_path='best_isic_model.pth'):
        """Initialize the enhanced analysis with CNN model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load optimal threshold if available
        self.optimal_threshold = 0.5  # Default threshold
        self.load_optimal_threshold()
        
        # Load CNN model if available
        self.load_cnn_model()
        
        # Skin type specific adjustments
        self.skin_type_adjustments = {
            'I': {'sensitivity': 1.2, 'specificity': 0.9},  # Very fair skin
            'II': {'sensitivity': 1.1, 'specificity': 0.95},  # Fair skin
            'III': {'sensitivity': 1.0, 'specificity': 1.0},  # Medium skin
            'IV': {'sensitivity': 0.9, 'specificity': 1.05},  # Olive skin
            'V': {'sensitivity': 0.8, 'specificity': 1.1},  # Dark skin
            'VI': {'sensitivity': 0.7, 'specificity': 1.15}  # Very dark skin
        }
        
        # Adjustment factors for darker skin (Fitzpatrick V/VI) - MUCH STRONGER
        self.DARKER_SKIN_BENIGN_CONFIDENCE_FACTOR = 0.5  # Reduce benign confidence by 50%
        self.DARKER_SKIN_MALIGNANT_CONFIDENCE_FACTOR = 1.4  # Increase malignant confidence by 40%
    
    def load_cnn_model(self):
        """Load the trained CNN model"""
        try:
            # Create ResNet-18 model with the same architecture as training
            model = models.resnet18(pretrained=False)
            
            # Replace final layer to match training architecture
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # 2 classes: benign, malignant
            )
            
            # Load trained weights
            if os.path.exists(self.model_path):
                model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                model.eval()
                self.cnn_model = model
                logger.info(f"CNN model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"CNN model not found at {self.model_path}. Using metadata-only analysis.")
                return False
            
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            return False
    
    def _create_custom_cnn(self):
        """Create a custom CNN architecture that matches the saved model"""
        class CustomCNN(nn.Module):
            def __init__(self, num_classes=1):  # Changed to 1 class for binary classification
                super(CustomCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, num_classes)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return CustomCNN(num_classes=1)
    
    def predict_with_cnn(self, image_path):
        """Predict using the CNN model"""
        try:
            if self.cnn_model is None:
                return None, 0.0
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.cnn_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction and confidence
                malignant_prob = probabilities[0][1].item()  # Probability of malignant class
                benign_prob = probabilities[0][0].item()     # Probability of benign class
                
                # Use optimal threshold for classification
                if malignant_prob > self.optimal_threshold:
                    prediction = "Malignant"
                    confidence = malignant_prob
                else:
                    prediction = "Benign"
                    confidence = benign_prob
                
                return prediction, confidence
            
        except Exception as e:
            logger.error(f"CNN prediction error: {e}")
            return None, 0.0
    
    def analyze_basic_features(self, image_path):
        """Analyze basic image features using PIL"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                total_pixels = width * height
                
                # Analyze color distribution
                stat = ImageStat.Stat(img)
                mean_colors = stat.mean
                std_colors = stat.stddev
                
                # Asymmetry analysis
                quarter_w, quarter_h = width // 2, height // 2
                top_left = img.crop((0, 0, quarter_w, quarter_h))
                top_right = img.crop((quarter_w, 0, width, quarter_h))
                bottom_left = img.crop((0, quarter_h, quarter_w, height))
                bottom_right = img.crop((quarter_w, quarter_h, width, height))
                
                tl_stat = ImageStat.Stat(top_left)
                tr_stat = ImageStat.Stat(top_right)
                bl_stat = ImageStat.Stat(bottom_left)
                br_stat = ImageStat.Stat(bottom_right)
                
                horizontal_diff = sum(abs(a - b) for a, b in zip(tl_stat.mean + bl_stat.mean, tr_stat.mean + br_stat.mean))
                vertical_diff = sum(abs(a - b) for a, b in zip(tl_stat.mean + tr_stat.mean, bl_stat.mean + br_stat.mean))
                asymmetry_score = (horizontal_diff + vertical_diff) / 6
                
                # Border irregularity
                edges = img.filter(ImageFilter.FIND_EDGES)
                edge_stat = ImageStat.Stat(edges)
                border_score = sum(edge_stat.mean) / 3
                
                # Color variation
                color_variation = sum(std_colors) / 3
                
                # Diameter estimation
                diameter_score = min(width, height) / 100
                
                return {
                    'asymmetry': min(asymmetry_score / 50, 1.0),
                    'border': min(border_score / 100, 1.0),
                    'color': min(color_variation / 50, 1.0),
                    'diameter': min(diameter_score, 1.0),
                    'mean_colors': mean_colors,
                    'std_colors': std_colors,
                    'size': (width, height)
                }
        except Exception as e:
            logger.error(f"Basic feature analysis error: {e}")
            return {
                'asymmetry': 0.3, 'border': 0.3, 'color': 0.3, 'diameter': 0.3,
                'mean_colors': [128, 128, 128], 'std_colors': [30, 30, 30], 'size': (100, 100)
            }
    
    def calculate_risk_factors(self, age, uv_exposure, family_history, skin_type, body_part, evolution_weeks):
        """Calculate additional risk factors"""
        risk_score = 0.0
        
        # Age risk
        if age > 50:
            risk_score += 0.2
        elif age > 65:
            risk_score += 0.3
        
        # UV exposure risk
        risk_score += min(uv_exposure / 10.0, 0.3)
        
        # Family history
        if family_history:
            risk_score += 0.2
        
        # Skin type risk
        skin_type_risk = {
            'I': 0.3, 'II': 0.25, 'III': 0.15,
            'IV': 0.1, 'V': 0.05, 'VI': 0.05
        }
        risk_score += skin_type_risk.get(skin_type, 0.15)
        
        # Body part risk
        high_risk_parts = ['trunk_back', 'trunk_chest', 'head_neck', 'shoulders']
        if body_part in high_risk_parts:
            risk_score += 0.15
        
        # Evolution risk
        if evolution_weeks > 0:
            risk_score += min(evolution_weeks / 52.0, 0.2)
        
        return min(risk_score, 1.0)
    
    def adjust_for_skin_type(self, prediction, confidence, skin_type):
        """Adjust predictions based on skin type"""
        if skin_type not in self.skin_type_adjustments:
            return prediction, confidence
        
        print(f"Adjusting for skin type: {skin_type}")
        adjustments = self.skin_type_adjustments[skin_type]
        adjusted_confidence = confidence * adjustments['sensitivity']
        
        # For darker skin types, be much more conservative with benign predictions
        if skin_type in ['V', 'VI']:
            print(f"Applying darker skin adjustments for type {skin_type}")
            if prediction == "Likely Benign - Routine Monitoring Recommended":
                adjusted_confidence *= self.DARKER_SKIN_BENIGN_CONFIDENCE_FACTOR
                print(f"Reduced benign confidence from {confidence:.1f}% to {adjusted_confidence:.1f}%")
            elif "Suspicious" in prediction or "Malignant" in prediction:
                adjusted_confidence *= self.DARKER_SKIN_MALIGNANT_CONFIDENCE_FACTOR
                print(f"Increased malignant confidence from {confidence:.1f}% to {adjusted_confidence:.1f}%")
        
        return prediction, min(adjusted_confidence, 1.0)
    
    def predict_lesion(self, image_path, skin_type='III', body_part='other', 
                      has_evolved=False, evolution_weeks=0, manual_length=None, 
                      manual_width=None, age=50, uv_exposure=5, family_history=False):
        """
        Enhanced prediction combining CNN and metadata analysis
        """
        try:
            # Detect skin tone from image
            detected_skin_type = self.detect_skin_type(image_path)
            print(f"Using skin type: {detected_skin_type} (detected from image)")
            
            # Get CNN prediction
            cnn_prediction, cnn_confidence = self.predict_with_cnn(image_path)
            print(f"CNN prediction: {cnn_prediction}, confidence: {cnn_confidence:.3f}")
            
            # Analyze basic features
            features = self.analyze_basic_features(image_path)
            
            # Calculate ABCDE scores
            asymmetry_score = features['asymmetry']
            border_score = features['border']
            color_score = features['color']
            diameter_score = features['diameter']
            evolution_score = 0.1 if has_evolved else 0.0
            
            # Weighted ABCDE score
            abcde_score = (
                asymmetry_score * 0.25 +
                border_score * 0.25 +
                color_score * 0.25 +
                diameter_score * 0.15 +
                evolution_score * 0.1
            )
            
            # Additional risk factors
            risk_factor_score = self.calculate_risk_factors(
                age, uv_exposure, family_history, detected_skin_type, body_part, evolution_weeks
            )
            
            # Combine metadata scores
            metadata_score = (abcde_score * 0.7) + (risk_factor_score * 0.3)
            
            # Combine CNN and metadata predictions
            if cnn_prediction is not None:
                # CNN is available - combine predictions
                cnn_malignant_prob = cnn_confidence if cnn_prediction == "Malignant" else (1 - cnn_confidence)
                
                # Weighted combination (70% CNN, 30% metadata)
                combined_score = (cnn_malignant_prob * 0.7) + (metadata_score * 0.3)
                
                # Adjust for skin type
                adjusted_prediction, adjusted_confidence = self.adjust_for_skin_type(
                    cnn_prediction, cnn_confidence, detected_skin_type
                )
                
                # Final prediction logic
                if combined_score > 0.6:
                    prediction = "Suspicious - Requires Medical Evaluation"
                    confidence = min(85 + (combined_score - 0.6) * 37.5, 95)
                elif combined_score > 0.4:
                    prediction = "Moderately Concerning - Monitor Closely"
                    confidence = 60 + (combined_score - 0.4) * 62.5
                else:
                    prediction = "Likely Benign - Routine Monitoring Recommended"
                    confidence = 70 + (0.4 - combined_score) * 75
                
                analysis_type = 'cnn_enhanced'
            else:
                # CNN not available - use metadata only
                if metadata_score > 0.6:
                    prediction = "Suspicious - Requires Medical Evaluation"
                    confidence = min(85 + (metadata_score - 0.6) * 37.5, 95)
                elif metadata_score > 0.4:
                    prediction = "Moderately Concerning - Monitor Closely"
                    confidence = 60 + (metadata_score - 0.4) * 62.5
                else:
                    prediction = "Likely Benign - Routine Monitoring Recommended"
                    confidence = 70 + (0.4 - metadata_score) * 75
                
                analysis_type = 'metadata_only'
            
            # Enhanced analysis for darker skin tones
            if detected_skin_type in ['V', 'VI']:
                analysis_type = f'{analysis_type}_darker_skin_optimized'
                print(f"Applying darker skin optimizations for type {detected_skin_type}")
                
                # Check for specific patterns in darker skin
                mean_colors = features['mean_colors']
                if mean_colors[0] < 100 and mean_colors[1] < 100 and mean_colors[2] < 100:
                    # Very dark lesion on dark skin - increase concern
                    confidence = min(confidence + 10, 95)
                    if prediction == "Likely Benign - Routine Monitoring Recommended":
                        prediction = "Moderately Concerning - Monitor Closely"
                        print("Upgraded prediction due to dark lesion on dark skin")
            
            print(f"Final prediction: {prediction}, confidence: {confidence:.1f}%")
            
            # Prepare comprehensive analysis data (ABCDEF first, then CNN, then meta)
            analysis_data = {
                # 1. ABCDE Feature Analysis
                'ABCDE_feature_analysis': {
                    'asymmetry': round(asymmetry_score, 2),
                    'border': round(border_score, 2),
                    'color': round(color_score, 2),
                    'diameter': round(diameter_score, 2),
                    'evolution': round(evolution_score, 2),
                    'explanation': (
                        "ABCDE features are clinical criteria for melanoma risk: "
                        "A=Asymmetry, B=Border irregularity, C=Color variation, D=Diameter, E=Evolution. "
                        "Higher scores indicate more concerning features."
                    )
                },
                # 2. CNN Analysis
                'cnn_analysis': {
                    'cnn_prediction': cnn_prediction,
                    'cnn_confidence': round(cnn_confidence, 3),
                    'explanation': (
                        "The CNN confidence score shows the percentage confidence in malignancy. "
                        "100% means high confidence the lesion is malignant; 0% means high confidence it is benign. "
                        "The prediction is based on deep learning analysis of the image."
                    )
                },
                # 3. Metadata & Risk Factors
                'metadata_risk_analysis': {
                    'risk_factors': round(risk_factor_score, 2),
                    'metadata_score': round(metadata_score, 2),
                    'explanation': (
                        "Risk factors include age, UV exposure, family history, skin type, body part, and lesion evolution. "
                        "The risk factor score summarizes these into a single value (higher = more risk). "
                        "The metadata score combines ABCDE and risk factors for a holistic risk estimate."
                    )
                },
                # 4. Combined Score
                'combined_score': round(combined_score if cnn_prediction is not None else metadata_score, 2),
                'combined_score_explanation': (
                    "The combined score is a weighted average of the CNN confidence and metadata score: "
                    "70% CNN, 30% metadata. Higher values indicate higher risk of malignancy."
                ),
                'detected_skin_tone': detected_skin_type,
                'analysis_type': analysis_type,
                'skin_type_adjustments': self.skin_type_adjustments.get(detected_skin_type, {}),
                'image_features': features,
                'manual_measurements': {
                    'length': manual_length,
                    'width': manual_width
                }
            }
            
            return prediction, round(confidence, 1), analysis_data
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return "Analysis Error - Please Consult Healthcare Provider", 50.0, {
                'error': str(e), 'analysis_type': 'error_fallback'
            }
    
    def load_optimal_threshold(self):
        """Load the optimal threshold for malignant class detection"""
        try:
            if os.path.exists('optimal_threshold.txt'):
                with open('optimal_threshold.txt', 'r') as f:
                    self.optimal_threshold = float(f.read().strip())
                print(f"Loaded optimal threshold: {self.optimal_threshold:.3f}")
            else:
                print("Optimal threshold file not found, using default threshold: 0.5")
        except Exception as e:
            print(f"Error loading optimal threshold: {e}, using default: 0.5")
    
    def detect_skin_type(self, image_path):
        """Estimate Fitzpatrick skin type from the image using border pixel brightness"""
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            border_pixels = []
            
            # Sample pixels from the border (top, bottom, left, right edges)
            for x in range(0, width, max(1, width//20)):  # Sample every 20th pixel
                border_pixels.append(img.getpixel((x, 0)))
                border_pixels.append(img.getpixel((x, height - 1)))
            for y in range(0, height, max(1, height//20)):  # Sample every 20th pixel
                border_pixels.append(img.getpixel((0, y)))
                border_pixels.append(img.getpixel((width - 1, y)))
            
            # Average brightness
            avg_rgb = tuple(sum(c) / len(border_pixels) for c in zip(*border_pixels))
            avg_brightness = sum(avg_rgb) / 3
            
            print(f"Skin tone detection - Average RGB: {avg_rgb}, Brightness: {avg_brightness:.1f}")
            
            # Map brightness to Fitzpatrick type with better thresholds
            if avg_brightness > 200:
                detected_type = 'I'
            elif avg_brightness > 160:
                detected_type = 'II'
            elif avg_brightness > 120:
                detected_type = 'III'
            elif avg_brightness > 80:
                detected_type = 'IV'
            elif avg_brightness > 50:
                detected_type = 'V'
            else:
                detected_type = 'VI'
            
            print(f"Detected skin type: {detected_type} (Fitzpatrick {detected_type})")
            return detected_type
            
        except Exception as e:
            print(f"Skin tone detection error: {e}")
            return 'III'  # Default to medium if uncertain

# Global instance
enhanced_analyzer = EnhancedSkinAnalysis()

def predict_lesion(image_path, skin_type='III', body_part='other', 
                  has_evolved=False, evolution_weeks=0, manual_length=None, 
                  manual_width=None, age=50, uv_exposure=5, family_history=False):
    """
    Main prediction function that uses the enhanced analyzer
    """
    return enhanced_analyzer.predict_lesion(
        image_path, skin_type, body_part, has_evolved, evolution_weeks,
        manual_length, manual_width, age, uv_exposure, family_history
    )