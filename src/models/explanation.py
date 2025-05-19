from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_explanation(subject: Dict, comparable: Dict, similarity_score: float) -> Tuple[List[str], float]:
    """
    Generate explanation for why a property is considered comparable.
    Returns a tuple of (explanation_points, confidence_score).
    """
    try:
        explanation_points = []
        confidence_score = 0.0
        total_weight = 0.0
        
        # Location comparison
        if 'latitude' in subject and 'longitude' in subject and \
           'latitude' in comparable and 'longitude' in comparable:
            loc_diff = calculate_location_difference(
                subject['latitude'], subject['longitude'],
                comparable['latitude'], comparable['longitude']
            )
            if loc_diff < 0.3:  # Within 3km
                explanation_points.append("Located in the same neighborhood")
                confidence_score += 0.3
                total_weight += 0.3
        
        # Price comparison
        if 'price' in subject and 'price' in comparable:
            price_diff = calculate_price_difference(
                subject['price'], comparable['price']
            )
            if price_diff < 0.2:  # Within 20% price difference
                explanation_points.append("Similar price point")
                confidence_score += 0.2
                total_weight += 0.2
        
        # Property characteristics comparison
        char_points, char_confidence = compare_characteristics(subject, comparable)
        explanation_points.extend(char_points)
        confidence_score += char_confidence
        total_weight += 0.5
        
        # Normalize confidence score
        if total_weight > 0:
            confidence_score = confidence_score / total_weight
        
        return explanation_points, confidence_score
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return [], 0.0

def calculate_location_difference(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate normalized difference in location."""
    try:
        # Haversine formula for distance calculation
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(lambda x: float(x) * 3.14159 / 180, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (dlat/2)**2 + (dlon/2)**2
        c = 2 * (a**0.5)
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

def compare_characteristics(subject: Dict, comparable: Dict) -> Tuple[List[str], float]:
    """Compare property characteristics and generate explanation points."""
    try:
        explanation_points = []
        confidence_score = 0.0
        total_weight = 0.0
        
        # GLA comparison
        if 'gla' in subject and 'gla' in comparable:
            gla_diff = abs(float(subject['gla']) - float(comparable['gla'])) / max(float(subject['gla']), float(comparable['gla']))
            if gla_diff < 0.2:  # Within 20% size difference
                explanation_points.append("Similar living area")
                confidence_score += 0.3
                total_weight += 0.3
        
        # Bedrooms comparison
        if 'bedrooms' in subject and 'bedrooms' in comparable:
            if subject['bedrooms'] == comparable['bedrooms']:
                explanation_points.append("Same number of bedrooms")
                confidence_score += 0.2
                total_weight += 0.2
        
        # Bathrooms comparison
        if 'bathrooms' in subject and 'bathrooms' in comparable:
            if subject['bathrooms'] == comparable['bathrooms']:
                explanation_points.append("Same number of bathrooms")
                confidence_score += 0.2
                total_weight += 0.2
        
        # Year built comparison
        if 'year_built' in subject and 'year_built' in comparable:
            year_diff = abs(int(subject['year_built']) - int(comparable['year_built']))
            if year_diff <= 5:
                explanation_points.append("Built in the same time period")
                confidence_score += 0.2
                total_weight += 0.2
        
        # Property type comparison
        if 'property_type' in subject and 'property_type' in comparable:
            if subject['property_type'] == comparable['property_type']:
                explanation_points.append("Same property type")
                confidence_score += 0.1
                total_weight += 0.1
        
        # Normalize confidence score
        if total_weight > 0:
            confidence_score = confidence_score / total_weight
        
        return explanation_points, confidence_score
        
    except Exception as e:
        logger.error(f"Error comparing characteristics: {str(e)}")
        return [], 0.0 