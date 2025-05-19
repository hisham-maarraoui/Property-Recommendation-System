import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_similarity(property1: Dict, property2: Dict) -> float:
    """
    Calculate similarity score between two properties based on various features.
    Returns a score between 0 and 1, where 1 means identical properties.
    """
    try:
        # Extract features
        features1 = extract_features(property1)
        features2 = extract_features(property2)
        
        if not features1 or not features2:
            return 0.0
        
        # Calculate feature differences
        differences = []
        weights = []
        
        # Location similarity (weight: 0.3)
        if 'latitude' in features1 and 'longitude' in features1 and \
           'latitude' in features2 and 'longitude' in features2:
            loc_diff = calculate_location_difference(
                features1['latitude'], features1['longitude'],
                features2['latitude'], features2['longitude']
            )
            differences.append(loc_diff)
            weights.append(0.3)
        
        # Price similarity (weight: 0.2)
        if 'price' in features1 and 'price' in features2:
            price_diff = calculate_price_difference(
                features1['price'], features2['price']
            )
            differences.append(price_diff)
            weights.append(0.2)
        
        # Property characteristics (weight: 0.5)
        char_diff = calculate_characteristics_difference(features1, features2)
        differences.append(char_diff)
        weights.append(0.5)
        
        # Calculate weighted average
        if differences and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
                similarity = 1 - sum(d * w for d, w in zip(differences, weights))
                return max(0.0, min(1.0, similarity))
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def extract_features(property_data: Dict) -> Dict:
    """Extract relevant features from property data."""
    features = {}
    
    # Location
    if 'latitude' in property_data and 'longitude' in property_data:
        features['latitude'] = float(property_data['latitude'])
        features['longitude'] = float(property_data['longitude'])
    
    # Price
    if 'price' in property_data:
        try:
            features['price'] = float(property_data['price'])
        except (ValueError, TypeError):
            pass
    
    # Property characteristics
    numeric_features = ['gla', 'bedrooms', 'bathrooms', 'year_built', 'lot_size']
    for feature in numeric_features:
        if feature in property_data:
            try:
                features[feature] = float(property_data[feature])
            except (ValueError, TypeError):
                pass
    
    # Property type
    if 'property_type' in property_data:
        features['property_type'] = property_data['property_type']
    
    return features

def calculate_location_difference(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate normalized difference in location."""
    try:
        # Haversine formula for distance calculation
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # Normalize distance (assuming 10km as max distance for comparison)
        max_distance = 10.0
        return min(1.0, distance / max_distance)
        
    except Exception as e:
        logger.error(f"Error calculating location difference: {str(e)}")
        return 1.0

def calculate_price_difference(price1: float, price2: float) -> float:
    """Calculate normalized difference in price."""
    try:
        if price1 <= 0 or price2 <= 0:
            return 1.0
        
        # Calculate percentage difference
        diff = abs(price1 - price2) / max(price1, price2)
        
        # Normalize difference (assuming 50% as max difference for comparison)
        max_diff = 0.5
        return min(1.0, diff / max_diff)
        
    except Exception as e:
        logger.error(f"Error calculating price difference: {str(e)}")
        return 1.0

def calculate_characteristics_difference(features1: Dict, features2: Dict) -> float:
    """Calculate normalized difference in property characteristics."""
    try:
        differences = []
        weights = []
        
        # GLA difference (weight: 0.3)
        if 'gla' in features1 and 'gla' in features2:
            gla_diff = abs(features1['gla'] - features2['gla']) / max(features1['gla'], features2['gla'])
            differences.append(min(1.0, gla_diff))
            weights.append(0.3)
        
        # Bedrooms difference (weight: 0.2)
        if 'bedrooms' in features1 and 'bedrooms' in features2:
            bed_diff = abs(features1['bedrooms'] - features2['bedrooms']) / max(features1['bedrooms'], features2['bedrooms'])
            differences.append(min(1.0, bed_diff))
            weights.append(0.2)
        
        # Bathrooms difference (weight: 0.2)
        if 'bathrooms' in features1 and 'bathrooms' in features2:
            bath_diff = abs(features1['bathrooms'] - features2['bathrooms']) / max(features1['bathrooms'], features2['bathrooms'])
            differences.append(min(1.0, bath_diff))
            weights.append(0.2)
        
        # Year built difference (weight: 0.2)
        if 'year_built' in features1 and 'year_built' in features2:
            year_diff = abs(features1['year_built'] - features2['year_built']) / 100  # Assuming 100 years as max difference
            differences.append(min(1.0, year_diff))
            weights.append(0.2)
        
        # Property type difference (weight: 0.1)
        if 'property_type' in features1 and 'property_type' in features2:
            type_diff = 1.0 if features1['property_type'] != features2['property_type'] else 0.0
            differences.append(type_diff)
            weights.append(0.1)
        
        # Calculate weighted average
        if differences and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
                return sum(d * w for d, w in zip(differences, weights))
        
        return 1.0
        
    except Exception as e:
        logger.error(f"Error calculating characteristics difference: {str(e)}")
        return 1.0 