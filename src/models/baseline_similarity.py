import numpy as np
from typing import Dict, List
import re

# Define the features to use for the baseline model
BASELINE_FEATURES = [
    'gla',            # Gross Living Area (should be numeric)
    'year_built',     # Year built (should be numeric)
    'num_beds',       # Number of bedrooms (should be numeric)
    'num_baths',      # Number of bathrooms (should be numeric or parsed)
    'condition',      # Condition (categorical)
]

# Simple mapping for condition to numeric
CONDITION_MAP = {
    'Excellent': 3,
    'Superior': 2,
    'Average': 1,
    'Inferior': 0,
    'Unknown': 1  # Default to Average for unknown conditions
}

# Weights for each feature in the similarity calculation
FEATURE_WEIGHTS = {
    'gla': 0.3,
    'year_built': 0.2,
    'num_beds': 0.2,
    'num_baths': 0.2,
    'condition': 0.1
}

def parse_bathrooms(bath_str):
    """Parse bath string like '1:1' into total baths (full + 0.5*half)."""
    if not bath_str or not isinstance(bath_str, str):
        return 0.0
    parts = bath_str.split(':')
    try:
        full = int(parts[0])
        half = int(parts[1]) if len(parts) > 1 else 0
        return full + 0.5 * half
    except Exception:
        return 0.0

def extract_features(property_data):
    """Extract relevant features from property data."""
    features = {}
    
    # Handle year_built - extract first number if it contains text or ranges
    year_built = property_data.get('year_built')
    if year_built:
        if isinstance(year_built, str):
            # Extract first number from string (e.g. "2012 +/-" -> 2012)
            match = re.search(r'\d+', str(year_built))
            if match:
                features['year_built'] = float(match.group())
            else:
                features['year_built'] = 0.0
        else:
            features['year_built'] = float(year_built)
    else:
        features['year_built'] = 0.0
    
    # Handle bedrooms - check multiple possible keys
    bedrooms = (
        property_data.get('num_bedrooms') or
        property_data.get('bedrooms') or
        property_data.get('num_beds')
    )
    if bedrooms is not None:
        if isinstance(bedrooms, str):
            match = re.search(r'\d+', bedrooms)
            features['num_beds'] = float(match.group()) if match else 0.0
        else:
            features['num_beds'] = float(bedrooms)
    else:
        features['num_beds'] = 0.0
    
    # Handle bathrooms - check multiple possible keys
    full_baths = property_data.get('full_baths')
    half_baths = property_data.get('half_baths')
    if full_baths is not None or half_baths is not None:
        full_baths = float(full_baths) if full_baths is not None else 0.0
        half_baths = float(half_baths) if half_baths is not None else 0.0
        features['num_baths'] = full_baths + (0.5 * half_baths)
    else:
        num_bathrooms = (
            property_data.get('num_bathrooms') or
            property_data.get('num_baths')
        )
        if num_bathrooms is not None:
            if isinstance(num_bathrooms, str):
                features['num_baths'] = parse_bathrooms(num_bathrooms)
            else:
                features['num_baths'] = float(num_bathrooms)
        else:
            features['num_baths'] = 0.0
    
    # Handle GLA (Gross Living Area) - map 'living_area' for subject, 'gla' for candidates
    gla = property_data.get('living_area', property_data.get('gla'))
    if gla:
        if isinstance(gla, str):
            # Remove any non-numeric characters except decimal point
            gla = ''.join(c for c in gla if c.isdigit() or c == '.')
            features['gla'] = float(gla) if gla else 0.0
        else:
            features['gla'] = float(gla)
    else:
        features['gla'] = 0.0
    
    # Handle condition
    condition = property_data.get('condition', 'Unknown')
    features['condition'] = CONDITION_MAP.get(condition, 1)  # Default to Average (1) if unknown
    
    return features

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to a 0-1 range."""
    # Avoid division by zero
    max_vals = np.maximum(np.max(features, axis=0), 1e-10)
    return features / max_vals

def compute_similarity(subject_features: Dict, candidate_features: Dict) -> float:
    """Compute similarity between subject and candidate properties."""
    # Debug: Print subject and candidate features
    print(f"Subject Features: {subject_features}")
    print(f"Candidate Features: {candidate_features}")
    
    # Normalize features
    subject_gla = subject_features['gla'] / 1000.0  # Convert to thousands of sq ft
    candidate_gla = candidate_features['gla'] / 1000.0
    
    # Compute weighted differences
    gla_diff = FEATURE_WEIGHTS['gla'] * ((subject_gla - candidate_gla) / max(subject_gla, 1.0)) ** 2
    year_diff = FEATURE_WEIGHTS['year_built'] * ((subject_features['year_built'] - candidate_features['year_built']) / 100.0) ** 2
    beds_diff = FEATURE_WEIGHTS['num_beds'] * ((subject_features['num_beds'] - candidate_features['num_beds']) / max(subject_features['num_beds'], 1.0)) ** 2
    baths_diff = FEATURE_WEIGHTS['num_baths'] * ((subject_features['num_baths'] - candidate_features['num_baths']) / max(subject_features['num_baths'], 1.0)) ** 2
    
    # Compute condition difference
    condition_diff = FEATURE_WEIGHTS['condition'] * ((subject_features['condition'] - candidate_features['condition']) / 3.0) ** 2
    
    # Total weighted distance
    distance = gla_diff + year_diff + beds_diff + baths_diff + condition_diff
    
    # Convert distance to similarity score (1.0 for identical, approaching 0 for very different)
    return 1.0 / (1.0 + distance)

def rank_candidates(subject: Dict, candidates: List[Dict], top_k=3) -> List[Dict]:
    """Rank candidates by similarity to subject and return top_k comps."""
    scored = []
    for cand in candidates:
        # Debug: Print candidate data before computing similarity
        print(f"Candidate Data: {cand}")
        # Ensure subject and candidate data are processed correctly
        subject_features = extract_features(subject)
        candidate_features = extract_features(cand)
        score = compute_similarity(subject_features, candidate_features)
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)  # Higher score = more similar
    return [cand for _, cand in scored[:top_k]] 