import json
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

class DataProcessor:
    def __init__(self, data_path: str):
        """Initialize the data processor with the path to the dataset."""
        self.data_path = data_path
        self.data = None
        self.processed_data = None

    def load_data(self) -> None:
        """Load the JSON dataset as a single object and extract the 'appraisals' list."""
        with open(self.data_path, 'r') as f:
            obj = json.load(f)
            self.data = obj.get('appraisals', [])

    def analyze_dataset(self) -> Dict:
        """Analyze the dataset and return basic statistics."""
        if not self.data:
            self.load_data()

        stats = {
            'total_appraisals': len(self.data),
            'features': {},
            'property_types': {},
            'avg_candidates_per_appraisal': 0,
            'avg_comps_per_appraisal': 0
        }

        # Analyze features and property types
        for appraisal in self.data:
            if not isinstance(appraisal, dict):
                continue
            subject = appraisal.get('subject', {})
            candidates = appraisal.get('properties', [])
            comps = appraisal.get('comps', [])

            if not isinstance(subject, dict):
                continue

            # Count property types
            prop_type = subject.get('property_type', 'unknown')
            stats['property_types'][prop_type] = stats['property_types'].get(prop_type, 0) + 1

            # Analyze features
            for feature, value in subject.items():
                if feature not in stats['features']:
                    stats['features'][feature] = {
                        'type': type(value).__name__,
                        'unique_values': set()
                    }
                stats['features'][feature]['unique_values'].add(str(value))

            stats['avg_candidates_per_appraisal'] += len(candidates)
            stats['avg_comps_per_appraisal'] += len(comps)

        # Calculate averages
        if stats['total_appraisals'] > 0:
            stats['avg_candidates_per_appraisal'] /= stats['total_appraisals']
            stats['avg_comps_per_appraisal'] /= stats['total_appraisals']

        # Convert sets to lengths for JSON serialization
        for feature in stats['features']:
            stats['features'][feature]['unique_values'] = len(stats['features'][feature]['unique_values'])

        return stats

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare the data for training the recommendation system."""
        if not self.data:
            self.load_data()

        training_data = []
        validation_data = []

        # Split data into training (80%) and validation (20%)
        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        split_idx = int(len(self.data) * 0.8)

        for idx in indices:
            appraisal = self.data[idx]
            if not isinstance(appraisal, dict):
                continue
                
            subject = appraisal.get('subject_property', {})
            candidates = appraisal.get('candidate_properties', [])
            selected_comps = appraisal.get('selected_comps', [])

            if not isinstance(subject, dict) or not isinstance(candidates, list):
                continue

            # Create positive and negative examples
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                    
                example = {
                    'subject_property': subject,
                    'candidate_property': candidate,
                    'is_comparable': candidate in selected_comps
                }
                
                if idx < split_idx:
                    training_data.append(example)
                else:
                    validation_data.append(example)

        return pd.DataFrame(training_data), pd.DataFrame(validation_data)

    def get_property_features(self, property_data: Dict) -> Dict:
        """Extract relevant features from a property."""
        if not isinstance(property_data, dict):
            return {}
            
        return {
            'gla': property_data.get('gross_living_area', 0),
            'lot_size': property_data.get('lot_size', 0),
            'bedrooms': property_data.get('bedrooms', 0),
            'bathrooms': property_data.get('bathrooms', 0),
            'year_built': property_data.get('year_built', 0),
            'sale_price': property_data.get('sale_price', 0),
            'sale_date': property_data.get('sale_date', ''),
            'location': property_data.get('location', {})
        }

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("appraisals_dataset.json")
    stats = processor.analyze_dataset()
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2)) 