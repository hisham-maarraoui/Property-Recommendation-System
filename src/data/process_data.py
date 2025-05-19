import json
import logging
from typing import List, Dict, Any
import random
from pathlib import Path
import re
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load appraisal data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'appraisals' in data:
                return data['appraisals']
            return []
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return []

def parse_bathroom_count(bath_str: str) -> float:
    """Parse bathroom count from various formats."""
    if not bath_str or not isinstance(bath_str, str):
        return 0.0
        
    try:
        # Handle formats like "2:1", "2F 1H", "2 Full/1Half"
        full_baths = 0
        half_baths = 0
        
        # Extract numbers and their types
        parts = re.findall(r'(\d+)([FH]|Full|Half)', bath_str)
        if parts:
            for num, bath_type in parts:
                if bath_type in ['F', 'Full']:
                    full_baths += int(num)
                elif bath_type in ['H', 'Half']:
                    half_baths += int(num)
        else:
            # Try format like "2:1"
            parts = bath_str.split(':')
            if len(parts) == 2:
                full_baths = int(parts[0])
                half_baths = int(parts[1])
            else:
                # Try to extract any number
                num = re.search(r'\d+', bath_str)
                if num:
                    full_baths = int(num.group())
        
        return full_baths + (half_baths * 0.5)
    except Exception as e:
        logger.error(f"Error parsing bathroom count '{bath_str}': {str(e)}")
        return 0.0

def generate_property_id(property_data: Dict[str, Any]) -> str:
    """Generate a unique property id if missing, using address and price."""
    id_val = property_data.get('id')
    if id_val is not None and str(id_val).strip() != '':
        return str(id_val)
    # Use address and price to generate a hash
    address = str(property_data.get('address', ''))
    price = str(property_data.get('sale_price', property_data.get('close_price', '0')))
    unique_str = address + '_' + price
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def normalize_property_data(property_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize property data to a consistent format."""
    try:
        # Extract numeric values from strings
        def extract_numeric(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if not isinstance(value, str):
                return 0.0
            # Remove currency symbols, commas, and non-numeric characters
            cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
            return float(cleaned) if cleaned else 0.0

        # Extract square footage from string
        def extract_sqft(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if not isinstance(value, str):
                return 0.0
            # Extract number before "SqFt" or similar
            parts = value.split()
            for part in parts:
                if part.replace('.', '').isdigit():
                    return float(part)
            return 0.0

        # Extract year from string
        def extract_year(value: Any) -> int:
            if isinstance(value, (int, float)):
                return int(value)
            if not isinstance(value, str):
                return 0
            # Try to extract a 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', value)
            if year_match:
                return int(year_match.group())
            return 0

        prop_id = generate_property_id(property_data)

        normalized = {
            'id': prop_id,
            'address': str(property_data.get('address', '')),
            'price': extract_numeric(property_data.get('sale_price', property_data.get('close_price', 0))),
            'size': extract_sqft(property_data.get('gla', 0)),
            'bedrooms': int(extract_numeric(property_data.get('num_beds', property_data.get('bedrooms', 0)))),
            'bathrooms': parse_bathroom_count(property_data.get('num_baths', property_data.get('bath_count', '0'))),
            'year_built': extract_year(property_data.get('year_built', 0)),
            'location': {
                'latitude': float(property_data.get('latitude', 0)),
                'longitude': float(property_data.get('longitude', 0))
            },
            'property_type': str(property_data.get('prop_type', property_data.get('property_sub_type', ''))),
            'condition': str(property_data.get('condition', '')),
            'basement': str(property_data.get('basement', property_data.get('basement_finish', ''))),
            'style': str(property_data.get('style', '')),
            'age': int(extract_numeric(property_data.get('age', property_data.get('effective_age', 0))))
        }
        
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing property data: {str(e)}")
        return {}

def prepare_test_data(appraisals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare test data from appraisals."""
    try:
        test_data = []
        
        for appraisal in appraisals:
            subject = normalize_property_data(appraisal['subject'])
            comps = [normalize_property_data(comp) for comp in appraisal['comps']]
            neighborhood_props = [normalize_property_data(prop) for prop in appraisal['properties']]

            # Ensure subject property is included in neighborhood_properties (by hash-based ID)
            subject_id = subject['id']
            neighborhood_ids = {prop['id'] for prop in neighborhood_props}
            if subject_id not in neighborhood_ids:
                neighborhood_props.append(subject)

            test_data.append({
                'target_property': subject,
                'comparable_properties': comps,
                'neighborhood_properties': neighborhood_props
            })
        
        return test_data
        
    except Exception as e:
        logger.error(f"Error preparing test data: {str(e)}")
        return []

def main():
    """Main data processing function."""
    try:
        # Create data directories
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('data/evaluation').mkdir(parents=True, exist_ok=True)
        
        # Load data
        appraisals = load_data('appraisals_dataset.json')
        if not appraisals:
            raise ValueError("No appraisals loaded from dataset")
        
        logger.info(f"Loaded {len(appraisals)} appraisals")
        
        # Prepare test data
        test_cases = prepare_test_data(appraisals)
        if not test_cases:
            raise ValueError("No test cases generated")
        
        # Save test data
        with open('data/processed/test_data.json', 'w') as f:
            json.dump(test_cases, f, indent=4)
        
        logger.info(f"Processed {len(test_cases)} test cases")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")

if __name__ == "__main__":
    main() 