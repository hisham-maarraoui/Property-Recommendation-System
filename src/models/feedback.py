import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackManager:
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_weights = {
            'gla': 0.25,
            'bedrooms': 0.15,
            'price': 0.20,
            'location': 0.20,
            'property_type': 0.10,
            'year_built': 0.05,
            'lot_size': 0.05
        }
    
    def _load_feedback(self) -> Dict:
        """Load existing feedback data from file."""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "feedback_entries": [],
                "model_updates": [],
                "feature_weights_history": []
            }
    
    def _save_feedback(self) -> None:
        """Save feedback data to file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
    
    def add_feedback(self, subject_id: str, comp_id: str, rating: int, 
                    comments: Optional[str] = None) -> bool:
        """Add new feedback entry."""
        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "subject_id": subject_id,
                "comp_id": comp_id,
                "rating": rating,
                "comments": comments
            }
            
            self.feedback_data["feedback_entries"].append(feedback_entry)
            self._save_feedback()
            
            # Update model if we have enough new feedback
            if len(self.feedback_data["feedback_entries"]) % 10 == 0:  # Update every 10 entries
                self.update_model()
            
            return True
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback data."""
        try:
            entries = self.feedback_data["feedback_entries"]
            if not entries:
                return {"total_entries": 0}
            
            ratings = [entry["rating"] for entry in entries]
            return {
                "total_entries": len(entries),
                "average_rating": np.mean(ratings),
                "rating_distribution": {
                    "1": len([r for r in ratings if r == 1]),
                    "2": len([r for r in ratings if r == 2]),
                    "3": len([r for r in ratings if r == 3]),
                    "4": len([r for r in ratings if r == 4]),
                    "5": len([r for r in ratings if r == 5])
                }
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            return {"total_entries": 0}
    
    def update_model(self, properties: List[Dict]) -> bool:
        """Update model based on feedback data."""
        try:
            if not self.feedback_data["feedback_entries"]:
                logger.debug("No feedback entries available.")
                return False
            
            # Prepare training data
            X = []
            y = []
            property_ids = [str(p.get("id")) for p in properties]
            logger.debug(f"Property IDs available: {property_ids}")
            logger.debug(f"Feedback entries: {self.feedback_data['feedback_entries']}")
            
            for entry in self.feedback_data["feedback_entries"]:
                subject = next((p for p in properties if str(p.get("id")) == str(entry["subject_id"])), None)
                comp = next((p for p in properties if str(p.get("id")) == str(entry["comp_id"])), None)
                
                if subject and comp:
                    # Calculate feature differences
                    features = []
                    for feature in self.feature_weights.keys():
                        subj_val = subject.get(feature, 0)
                        comp_val = comp.get(feature, 0)
                        try:
                            subj_val = float(subj_val) if subj_val is not None else 0.0
                        except (ValueError, TypeError):
                            subj_val = 0.0
                        try:
                            comp_val = float(comp_val) if comp_val is not None else 0.0
                        except (ValueError, TypeError):
                            comp_val = 0.0
                        diff = abs(subj_val - comp_val)
                        features.append(diff)
                    X.append(features)
                    y.append(entry["rating"])
                else:
                    logger.debug(f"Could not find subject or comp for entry: {entry}")
            
            logger.debug(f"Training data X size: {len(X)}, y size: {len(y)}")
            if not X or not y:
                logger.debug("No valid training data generated.")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Update feature weights based on model importance
            feature_importance = self.model.feature_importances_
            total_importance = sum(feature_importance)
            
            if total_importance > 0:
                new_weights = {
                    feature: float(importance / total_importance)
                    for feature, importance in zip(self.feature_weights.keys(), feature_importance)
                }
                
                # Record weight update
                self.feedback_data["feature_weights_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "old_weights": self.feature_weights.copy(),
                    "new_weights": new_weights
                })
                
                # Update weights
                self.feature_weights = new_weights
            
            # Record model update
            self.feedback_data["model_updates"].append({
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(X),
                "feature_weights": self.feature_weights
            })
            
            self._save_feedback()
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return False
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get current feature weights."""
        return self.feature_weights.copy()
    
    def get_weight_history(self) -> List[Dict]:
        """Get history of feature weight updates."""
        return self.feedback_data["feature_weights_history"] 