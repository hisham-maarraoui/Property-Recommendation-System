from typing import Dict, List, Optional
import logging
import numpy as np
from data.processor import PropertyDataProcessor
from models.similarity import calculate_similarity
from models.explanation import generate_explanation
from models.feedback import FeedbackManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyRecommender:
    def __init__(self):
        self.processor = PropertyDataProcessor()
        self.feedback_manager = FeedbackManager()
        self.properties = []
    
    def load_properties(self, file_path: str) -> bool:
        """Load and preprocess all properties from the dataset."""
        try:
            raw_data = self.processor.load_data(file_path)
            self.properties = [
                self.processor.preprocess_property(prop)
                for prop in raw_data
                if prop.get('id') is not None
            ]
            
            # Train the explainer on the loaded properties
            self.explainer.train(self.properties)
            
            # Update model with feedback data
            self.feedback_manager.update_model(self.properties)
            
            # Update processor weights with feedback-based weights
            self.processor.feature_weights = self.feedback_manager.get_feature_weights()
            
            logger.info(f"Loaded {len(self.properties)} properties")
            return True
        except Exception as e:
            logger.error(f"Error loading properties: {str(e)}")
            return False
    
    def get_recommendations(self, subject_property: Dict, top_n: int = 3) -> List[Dict]:
        """Get top N comparable properties for a subject property."""
        try:
            processed_subject = self.processor.preprocess_property(subject_property)
            if not processed_subject:
                logger.error("Failed to preprocess subject property")
                return []
            if not self.properties:
                logger.error("No properties available for comparison")
                return []
            similarities = []
            for candidate in self.properties:
                if candidate.get('id') == processed_subject.get('id'):
                    continue
                if not candidate.get('address') or not candidate.get('price'):
                    continue
                similarity = self.processor.calculate_property_similarity(processed_subject, candidate)
                if similarity > 0.1:
                    candidate['similarity_score'] = similarity
                    similarities.append(candidate)
            similarities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            recommendations = similarities[:top_n]
            for rec in recommendations:
                explanation, confidence = generate_explanation(processed_subject, rec, rec.get('similarity_score', 0))
                rec['explanation'] = explanation
                rec['explanation_confidence'] = confidence
                rec['feature_importance'] = {}  # Placeholder, can be filled with actual importance if available
            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def add_feedback(self, subject_id: str, comp_id: str, rating: int, 
                    comments: Optional[str] = None) -> bool:
        """Add feedback for a recommendation."""
        try:
            success = self.feedback_manager.add_feedback(subject_id, comp_id, rating, comments)
            if success:
                # Update model with new feedback
                self.feedback_manager.update_model(self.properties)
                # Update processor weights
                self.processor.feature_weights = self.feedback_manager.get_feature_weights()
            return success
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback data."""
        return self.feedback_manager.get_feedback_stats()
    
    def get_weight_history(self) -> List[Dict]:
        """Get history of feature weight updates."""
        return self.feedback_manager.get_weight_history()
    
    def get_property_by_id(self, property_id: int) -> Optional[Dict]:
        """Get a property by its ID."""
        try:
            for prop in self.properties:
                if prop['id'] == property_id:
                    return prop
            return None
        except Exception as e:
            logger.error(f"Error getting property by ID: {str(e)}")
            return None 