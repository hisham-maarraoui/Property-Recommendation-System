import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyExplainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'gla', 'bedrooms', 'bathrooms', 'price', 'year_built',
            'lot_size', 'latitude', 'longitude', 'days_since_close'
        ]
        self.explainer = None
        self.is_trained = False

    def _prepare_features(self, properties: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert property dictionaries to feature matrix and target vector."""
        X = []
        y = []
        
        for prop in properties:
            features = []
            for feature in self.feature_names:
                value = prop.get(feature, 0)
                if isinstance(value, (int, float)):
                    features.append(value)
                else:
                    features.append(0)
            X.append(features)
            y.append(prop.get('price', 0))
        
        return np.array(X), np.array(y)

    def train(self, properties: List[Dict]) -> None:
        """Train the explainer model on property data."""
        try:
            # Prepare data
            X, y = self._prepare_features(properties)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            self.is_trained = True
            
            logger.info("Successfully trained property explainer")
        except Exception as e:
            logger.error(f"Error training explainer: {str(e)}")
            self.is_trained = False

    def explain_comparison(self, subject: Dict, comp: Dict) -> Dict:
        """Generate detailed explanation for why a property is comparable."""
        try:
            if not self.is_trained:
                return {"error": "Explainer not trained"}

            # Prepare feature vectors
            subject_features = self._prepare_features([subject])[0]
            comp_features = self._prepare_features([comp])[0]
            
            # Scale features
            subject_scaled = self.scaler.transform(subject_features.reshape(1, -1))
            comp_scaled = self.scaler.transform(comp_features.reshape(1, -1))
            
            # Get SHAP values
            subject_shap = self.explainer.shap_values(subject_scaled)[0]
            comp_shap = self.explainer.shap_values(comp_scaled)[0]
            
            # Calculate feature importance for comparison
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                importance = abs(subject_shap[i] - comp_shap[i])
                feature_importance[feature] = importance
            
            # Generate explanations for top features
            explanations = []
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for feature, importance in top_features:
                if feature == 'gla':
                    diff = comp['gla'] - subject['gla']
                    pct_diff = abs(diff) / max(subject['gla'], 1) * 100
                    if pct_diff < 10:
                        explanations.append(f"Very similar living area ({abs(diff):.0f} sq ft difference)")
                    else:
                        explanations.append(f"{'Larger' if diff > 0 else 'Smaller'} living area by {abs(diff):.0f} sq ft")
                
                elif feature == 'price':
                    diff = comp['price'] - subject['price']
                    pct_diff = abs(diff) / max(subject['price'], 1) * 100
                    if pct_diff < 10:
                        explanations.append(f"Very similar price point (${abs(diff):,.0f} difference)")
                    else:
                        explanations.append(f"{'Higher' if diff > 0 else 'Lower'} price by ${abs(diff):,.0f}")
                
                elif feature == 'bedrooms':
                    diff = comp['bedrooms'] - subject['bedrooms']
                    if diff == 0:
                        explanations.append("Same number of bedrooms")
                    else:
                        explanations.append(f"{'One more' if diff > 0 else 'One fewer'} bedroom")
                
                elif feature == 'year_built':
                    diff = comp['year_built'] - subject['year_built']
                    if abs(diff) <= 5:
                        explanations.append("Very similar construction year")
                    else:
                        explanations.append(f"{'Newer' if diff > 0 else 'Older'} by {abs(diff)} years")
                
                elif feature == 'lot_size':
                    diff = comp['lot_size'] - subject['lot_size']
                    pct_diff = abs(diff) / max(subject['lot_size'], 1) * 100
                    if pct_diff < 10:
                        explanations.append("Very similar lot size")
                    else:
                        explanations.append(f"{'Larger' if diff > 0 else 'Smaller'} lot by {abs(diff):.0f} sq ft")
            
            # Add location-based explanation
            if 'latitude' in feature_importance and 'longitude' in feature_importance:
                explanations.append("Similar location characteristics")
            
            # Calculate overall similarity score
            similarity_score = 1 - (sum(feature_importance.values()) / len(feature_importance))
            
            return {
                "explanations": explanations,
                "similarity_score": similarity_score,
                "feature_importance": feature_importance,
                "confidence": self._calculate_confidence(feature_importance)
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {"error": str(e)}

    def _calculate_confidence(self, feature_importance: Dict[str, float]) -> float:
        """Calculate confidence score for the explanation."""
        try:
            # Normalize feature importance scores
            total_importance = sum(feature_importance.values())
            if total_importance == 0:
                return 0.0
            
            normalized_importance = {k: v/total_importance for k, v in feature_importance.items()}
            
            # Calculate entropy-based confidence
            entropy = -sum(p * np.log2(p) for p in normalized_importance.values() if p > 0)
            max_entropy = np.log2(len(normalized_importance))
            
            # Convert entropy to confidence (higher entropy = lower confidence)
            confidence = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            
            return confidence
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0 