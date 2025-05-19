import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import math
import hashlib

logger = logging.getLogger(__name__)

class PropertyModel:
    def __init__(self, config_path: str = "config/model_config.json"):
        """Initialize the property recommendation model."""
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.property_features = {}
        self.property_embeddings = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {
                "n_clusters": 5,
                "similarity_threshold": 0.7,
                "max_recommendations": 5,
                "feature_weights": {
                    "price": 0.3,
                    "size": 0.2,
                    "location": 0.2,
                    "bedrooms": 0.1,
                    "bathrooms": 0.1,
                    "year_built": 0.1
                }
            }
    
    def _generate_property_id(self, property_data: Dict[str, Any]) -> str:
        """Generate a unique property id using address and price."""
        try:
            # First check if there's an existing ID
            if 'id' in property_data and property_data['id']:
                return str(property_data['id'])
            if 'property_id' in property_data and property_data['property_id']:
                return str(property_data['property_id'])
            
            # If no existing ID, generate one from address and price
            address = str(property_data.get('address', '')).strip()
            price = str(property_data.get('price', property_data.get('sale_price', property_data.get('close_price', '0'))))
            
            if not address or not price:
                # If missing critical data, generate a random ID
                return f"prop_{hash(str(property_data))}"
            
            unique_str = f"{address}_{price}"
            # Generate a shorter ID to match the format in test data
            return str(hash(unique_str) % 1000000)  # Generate a 6-digit numeric ID
        except Exception as e:
            logger.error(f"Error generating property ID: {str(e)}")
            return f"prop_{hash(str(property_data))}"
    
    def _extract_features(self, property_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize features from property data."""
        try:
            # Helper function to safely extract numeric values
            def safe_float(value, default=0.0):
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    # Remove currency symbols and commas
                    cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                    return float(cleaned) if cleaned else default
                return default

            # Basic features with safe extraction
            features = {
                'price': safe_float(property_data.get('price', property_data.get('sale_price', property_data.get('close_price', 0)))),
                'size': safe_float(property_data.get('size', property_data.get('gla', 0))),
                'bedrooms': safe_float(property_data.get('bedrooms', property_data.get('num_beds', 0))),
                'bathrooms': safe_float(property_data.get('bathrooms', property_data.get('num_baths', 0))),
                'year_built': safe_float(property_data.get('year_built', 0))
            }
            
            # Location features with validation
            location = property_data.get('location', {})
            lat = safe_float(location.get('latitude', 0))
            lon = safe_float(location.get('longitude', 0))
            
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                lat, lon = 0, 0
            
            features['latitude'] = lat
            features['longitude'] = lon
            
            # Calculate derived features
            current_year = datetime.now().year
            features['age'] = max(0, current_year - features['year_built'])
            
            # Calculate price per square foot (avoid division by zero)
            features['price_per_sqft'] = features['price'] / max(features['size'], 1)
            
            # Calculate room ratio (avoid division by zero)
            total_rooms = features['bedrooms'] + features['bathrooms']
            features['room_ratio'] = features['size'] / max(total_rooms, 1)
            
            # Log feature extraction
            logger.debug(f"Extracted features for property {property_data.get('id')}: {features}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return default features instead of empty dict
            return {
                'price': 0.0,
                'size': 0.0,
                'bedrooms': 0.0,
                'bathrooms': 0.0,
                'year_built': 0.0,
                'latitude': 0.0,
                'longitude': 0.0,
                'age': 0.0,
                'price_per_sqft': 0.0,
                'room_ratio': 0.0
            }
    
    def _calculate_similarity(self, prop1: Dict[str, float], prop2: Dict[str, float]) -> float:
        """Calculate similarity between two properties using weighted features."""
        try:
            weights = self.config['feature_weights']
            similarities = []
            
            # Price similarity (using relative difference)
            max_price = max(prop1['price'], prop2['price'])
            if max_price > 0:
                price_diff = abs(prop1['price'] - prop2['price']) / max_price
                # Use a more lenient price difference threshold
                price_sim = 1 - min(price_diff * 2, 1.0)  # Allow up to 50% price difference
                similarities.append(price_sim * weights['price'])
            
            # Size similarity
            max_size = max(prop1['size'], prop2['size'])
            if max_size > 0:
                size_diff = abs(prop1['size'] - prop2['size']) / max_size
                # Use a more lenient size difference threshold
                size_sim = 1 - min(size_diff * 2, 1.0)  # Allow up to 50% size difference
                similarities.append(size_sim * weights['size'])
            
            # Location similarity (using Haversine distance)
            loc_sim = self._calculate_location_similarity(
                prop1['latitude'], prop1['longitude'],
                prop2['latitude'], prop2['longitude']
            )
            similarities.append(loc_sim * weights['location'])
            
            # Room similarity (more lenient for small differences)
            bed_diff = abs(prop1['bedrooms'] - prop2['bedrooms'])
            room_sim = 1 - min(bed_diff / 2, 1.0)  # Allow up to 2 bedroom difference
            similarities.append(room_sim * weights['bedrooms'])
            
            # Bathroom similarity (more lenient for small differences)
            bath_diff = abs(prop1['bathrooms'] - prop2['bathrooms'])
            bath_sim = 1 - min(bath_diff, 1.0)  # Allow up to 1 bathroom difference
            similarities.append(bath_sim * weights['bathrooms'])
            
            # Age similarity (more lenient for older properties)
            age_diff = abs(prop1['age'] - prop2['age'])
            max_age = max(prop1['age'], prop2['age'])
            if max_age > 0:
                # Allow larger age differences for older properties
                age_sim = 1 - min(age_diff / (max_age + 10), 1.0)
            else:
                age_sim = 1.0
            similarities.append(age_sim * weights['year_built'])
            
            # Calculate weighted average
            if similarities:
                total_similarity = sum(similarities) / sum(weights.values())
                # Apply a minimum similarity threshold
                return max(total_similarity, 0.1)  # Ensure some minimum similarity
            return 0.1  # Default minimum similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.1  # Return minimum similarity on error
    
    def _calculate_location_similarity(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate location similarity using Haversine distance."""
        try:
            # Handle invalid coordinates
            if not all(isinstance(x, (int, float)) for x in [lat1, lon1, lat2, lon2]):
                return 0.5  # Default similarity for invalid coordinates
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371 * c  # Earth's radius in km
            
            # Use a more lenient distance threshold (20km instead of 10km)
            max_distance = 20
            similarity = 1 - min(distance / max_distance, 1.0)
            
            # Ensure minimum similarity
            return max(similarity, 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating location similarity: {str(e)}")
            return 0.5  # Default similarity on error
    
    def train(self, properties: List[Dict[str, Any]]) -> None:
        """Train the model on the property dataset."""
        try:
            # Extract features for all properties
            features_list = []
            valid_properties = []
            for prop in properties:
                # Generate consistent ID for each property
                prop['id'] = self._generate_property_id(prop)
                features = self._extract_features(prop)
                if features:
                    self.property_features[prop['id']] = features
                    features_list.append(list(features.values()))
                    valid_properties.append(prop)
            
            if not features_list:
                raise ValueError("No valid features extracted from properties")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_list)
            
            # Adjust number of clusters based on data size
            n_clusters = min(self.config['n_clusters'], len(features_list))
            if n_clusters < 2:
                n_clusters = 2  # Minimum 2 clusters
            
            # Train clustering model
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42
            )
            self.cluster_model.fit(scaled_features)
            
            # Store embeddings
            for i, prop in enumerate(valid_properties):
                if prop['id'] in self.property_features:
                    self.property_embeddings[prop['id']] = scaled_features[i]
            
            logger.info(f"Model trained on {len(valid_properties)} properties")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def find_similar_properties(self, property_id: str, n: int = None) -> List[Dict[str, Any]]:
        """Find similar properties using enhanced similarity metrics."""
        try:
            if n is None:
                n = self.config['max_recommendations']
            
            if property_id not in self.property_features:
                raise ValueError(f"Property {property_id} not found in trained model")
            
            # Get target property features
            target_features = self.property_features[property_id]
            target_embedding = self.property_embeddings[property_id]
            
            # Calculate similarities
            similarities = []
            for pid, features in self.property_features.items():
                if pid != property_id:
                    # Calculate feature-based similarity
                    feature_sim = self._calculate_similarity(target_features, features)
                    
                    # Calculate embedding-based similarity
                    embedding_sim = cosine_similarity(
                        target_embedding.reshape(1, -1),
                        self.property_embeddings[pid].reshape(1, -1)
                    )[0][0]
                    
                    # Combine similarities
                    combined_sim = 0.7 * feature_sim + 0.3 * embedding_sim
                    
                    similarities.append((pid, combined_sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N similar properties with their IDs
            return [{'id': pid, 'property': self.property_features[pid], 'similarity_score': score} 
                   for pid, score in similarities[:n]]
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {str(e)}")
            return []
    
    def rank_properties(self, properties: List[Dict[str, Any]], 
                       target_property: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank properties based on similarity to target property."""
        try:
            # Extract features for target property
            target_features = self._extract_features(target_property)
            
            # Calculate similarities and rank
            ranked_properties = []
            for prop in properties:
                features = self._extract_features(prop)
                similarity = self._calculate_similarity(target_features, features)
                
                ranked_properties.append({
                    'property': prop,
                    'similarity_score': similarity
                })
            
            # Sort by similarity score
            ranked_properties.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return ranked_properties
            
        except Exception as e:
            logger.error(f"Error ranking properties: {str(e)}")
            return [] 