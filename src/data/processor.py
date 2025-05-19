import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import re
from Levenshtein import distance
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyDataProcessor:
    def __init__(self):
        self.feature_weights = {
            'gla': 0.25,
            'bedrooms': 0.15,
            'price': 0.20,
            'location': 0.20,
            'property_type': 0.10,
            'year_built': 0.05,
            'lot_size': 0.05
        }
        self.abbreviations = {
            'st': 'street',
            'ave': 'avenue',
            'rd': 'road',
            'blvd': 'boulevard',
            'dr': 'drive',
            'ln': 'lane',
            'ct': 'court',
            'crt': 'court',
            'pl': 'place',
            'cres': 'crescent',
            'cir': 'circle',
            'sq': 'square',
            'ter': 'terrace',
            'trl': 'trail',
            'wy': 'way',
            'hwy': 'highway',
            'pkwy': 'parkway',
            'unit': '',
            'apt': '',
            'suite': '',
            '#': '',
            'n': 'north',
            's': 'south',
            'e': 'east',
            'w': 'west',
            'ne': 'northeast',
            'nw': 'northwest',
            'se': 'southeast',
            'sw': 'southwest'
        }
        
        # Common address patterns to clean
        self.address_patterns = [
            (r'\b(unit|apt|suite|#)\s*\d+[\s\-:,]*', ''),
            (r'\b\d+\s*(st|nd|rd|th)\b', lambda m: m.group(0).replace('st', 'street').replace('nd', 'street').replace('rd', 'street').replace('th', 'street')),
            (r'[^\w\s]', ' '),
            (r'\s+', ' ')
        ]

    def _generate_property_id(self, property_data: Dict) -> str:
        """Generate a consistent property ID from address and other unique identifiers."""
        try:
            # Use address as primary identifier
            address = str(property_data.get('address', '')).strip()
            if not address:
                return None
            
            # Add other unique identifiers if available
            identifiers = [address]
            if 'id' in property_data:
                identifiers.append(str(property_data['id']))
            if 'property_id' in property_data:
                identifiers.append(str(property_data['property_id']))
            
            # Create a hash of the combined identifiers
            unique_str = '_'.join(identifiers)
            return str(hash(unique_str) % 1000000)  # Generate a 6-digit numeric ID
        except Exception as e:
            logger.error(f"Error generating property ID: {str(e)}")
            return None

    def load_data(self, file_path: str) -> List[Dict]:
        """Load property data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data.get('appraisals', [])
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
    
    def _normalize_address(self, address: str) -> str:
        """Enhanced address normalization with better pattern matching and cleaning."""
        if not address:
            return ''
        
        # Convert to lowercase and remove extra spaces
        address = address.lower().strip()
        
        # Apply address patterns
        for pattern, replacement in self.address_patterns:
            if callable(replacement):
                address = re.sub(pattern, replacement, address)
            else:
                address = re.sub(pattern, replacement, address)
        
        # Replace abbreviations
        for abbr, full in self.abbreviations.items():
            address = re.sub(r'\b' + abbr + r'\b', full, address)
        
        # Remove common noise words
        noise_words = ['near', 'close to', 'across from', 'behind', 'in front of']
        for word in noise_words:
            address = address.replace(word, '')
        
        # Final cleanup
        address = ' '.join(address.split())
        return address

    def _calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity between two addresses using multiple methods."""
        if not addr1 or not addr2:
            return 0.0
        
        # Normalize addresses
        addr1 = self._normalize_address(addr1)
        addr2 = self._normalize_address(addr2)
        
        # Calculate Levenshtein distance similarity
        lev_sim = 1 - (distance(addr1, addr2) / max(len(addr1), len(addr2)))
        
        # Calculate sequence similarity
        seq_sim = SequenceMatcher(None, addr1, addr2).ratio()
        
        # Calculate word overlap
        words1 = set(addr1.split())
        words2 = set(addr2.split())
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        # Combine similarities with weights
        return 0.4 * lev_sim + 0.3 * seq_sim + 0.3 * word_overlap

    def _extract_price(self, data: Dict) -> float:
        """Enhanced price extraction with better validation and cleaning."""
        price_fields = ['price', 'close_price', 'sale_price', 'list_price', 'asking_price']
        
        for field in price_fields:
            if field in data and data[field]:
                try:
                    price_str = str(data[field])
                    logger.info(f"Checking field '{field}' with value: {price_str}")
                    # Remove currency symbols and commas
                    price_str = re.sub(r'[^\d.]', '', price_str)
                    price = float(price_str)
                    
                    # Basic validation
                    if 1000 <= price <= 1000000000:  # Reasonable price range
                        logger.info(f"Valid price found: {price}")
                        return price
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert '{field}' value to float: {price_str}")
                    continue
        
        logger.warning("No valid price found in the data.")
        return 0.0

    def _extract_numeric_feature(self, data: Dict, fields: List[str], default: float = 0.0) -> float:
        """Extract and clean numeric features from multiple possible field names."""
        for field in fields:
            if field in data and data[field]:
                try:
                    value = str(data[field])
                    # Remove non-numeric characters except decimal point
                    cleaned = re.sub(r'[^\d.]', '', value)
                    if cleaned:
                        return float(cleaned)
                except (ValueError, TypeError):
                    continue
        return default

    def _extract_bathrooms(self, data: Dict) -> float:
        """Extract and clean bathroom count, handling formats like '2:1'."""
        # Try full_baths and half_baths first
        full_baths = self._extract_numeric_feature(data, ['full_baths'], 0.0)
        half_baths = self._extract_numeric_feature(data, ['half_baths'], 0.0)
        if full_baths > 0 or half_baths > 0:
            return full_baths + (half_baths * 0.5)
        
        # Try other bathroom fields
        for field in ['bathrooms', 'num_baths', 'baths', 'bath_count']:
            if field in data and data[field]:
                try:
                    value = str(data[field])
                    # Handle format like "2:1"
                    if ':' in value:
                        full, half = value.split(':')
                        return float(full) + (float(half) * 0.5)
                    # Handle regular numeric format
                    cleaned = re.sub(r'[^\d.]', '', value)
                    if cleaned:
                        return float(cleaned)
                except (ValueError, TypeError):
                    continue
        return 0.0

    def preprocess_property(self, property_data: Dict) -> Dict:
        """Enhanced property preprocessing with better feature extraction and validation."""
        try:
            # Generate consistent property ID
            property_id = self._generate_property_id(property_data)
            if not property_id:
                logger.warning(f"Could not generate property ID for property: {property_data}")
                return None

            # Process the property data with enhanced feature extraction
            processed = {
                'id': property_id,
                'address': self._normalize_address(str(property_data.get('address', '')).strip()),
                'city': str(property_data.get('city', '')).strip().lower(),
                'province': str(property_data.get('province', '')).strip().lower(),
                'postal_code': str(property_data.get('postal_code', '')).strip().lower(),
                'latitude': self._extract_numeric_feature(property_data, ['latitude', 'lat']),
                'longitude': self._extract_numeric_feature(property_data, ['longitude', 'long', 'lng']),
                'bedrooms': int(self._extract_numeric_feature(property_data, ['bedrooms', 'num_beds', 'beds', 'bed_count'])),
                'bathrooms': self._extract_bathrooms(property_data),
                'gla': self._extract_numeric_feature(property_data, ['gla', 'size', 'square_feet', 'sqft']),
                'lot_size': self._extract_numeric_feature(property_data, ['lot_size', 'lot_sqft', 'land_size', 'lot_size_sf']),
                'price': self._extract_price(property_data),
                'year_built': int(self._extract_numeric_feature(property_data, ['year_built', 'construction_year'])),
                'property_type': str(property_data.get('property_type', property_data.get('structure_type', property_data.get('prop_type', '')))).strip().lower(),
                'style': str(property_data.get('style', '')).strip().lower(),
                'close_date': str(property_data.get('close_date', '')).strip(),
                'description': str(property_data.get('description', property_data.get('public_remarks', ''))).strip()
            }

            # Calculate days since close date
            if processed['close_date']:
                try:
                    close_date = datetime.strptime(processed['close_date'], '%Y-%m-%d')
                    processed['days_since_close'] = (datetime.now() - close_date).days
                except:
                    processed['days_since_close'] = 0
            else:
                processed['days_since_close'] = 0

            # Add derived features
            if processed['gla'] and processed['price']:
                processed['price_per_sqft'] = processed['price'] / processed['gla']
            else:
                processed['price_per_sqft'] = 0.0

            # Ensure all features used in feedback/modeling are numeric
            numeric_features = [
                'gla', 'bedrooms', 'price', 'location', 'year_built', 'lot_size'
            ]
            for feature in numeric_features:
                val = processed.get(feature, 0)
                try:
                    processed[feature] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    processed[feature] = 0.0

            return processed
        except Exception as e:
            logger.error(f"Error preprocessing property: {str(e)}")
            return None
    
    def calculate_location_similarity(self, prop1: Dict, prop2: Dict) -> float:
        """Calculate location similarity using Haversine distance."""
        try:
            from math import radians, sin, cos, sqrt, atan2
            
            lat1, lon1 = radians(prop1['latitude']), radians(prop1['longitude'])
            lat2, lon2 = radians(prop2['latitude']), radians(prop2['longitude'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = 6371 * c  # Earth's radius in km
            
            # Convert distance to similarity score (closer = more similar)
            similarity = 1 / (1 + distance)
            return similarity
        except Exception as e:
            logger.error(f"Error calculating location similarity: {str(e)}")
            return 0.0
    
    def calculate_property_similarity(self, subject: Dict, candidate: Dict) -> float:
        """Enhanced property similarity calculation with more features and better weighting."""
        try:
            similarities = {}
            
            # GLA similarity with better normalization
            if subject['gla'] and candidate['gla']:
                gla_diff = abs(subject['gla'] - candidate['gla'])
                similarities['gla'] = 1 / (1 + gla_diff/max(subject['gla'], candidate['gla']))
            else:
                similarities['gla'] = 0
            
            # Bedroom similarity
            if subject['bedrooms'] and candidate['bedrooms']:
                bedroom_diff = abs(subject['bedrooms'] - candidate['bedrooms'])
                similarities['bedrooms'] = 1 / (1 + bedroom_diff)
            else:
                similarities['bedrooms'] = 0
            
            # Price similarity with better normalization
            if subject['price'] and candidate['price']:
                price_diff = abs(subject['price'] - candidate['price'])
                similarities['price'] = 1 / (1 + price_diff/max(subject['price'], candidate['price']))
            else:
                similarities['price'] = 0
            
            # Location similarity
            similarities['location'] = self.calculate_location_similarity(subject, candidate)
            
            # Property type similarity
            similarities['property_type'] = 1.0 if subject['property_type'] == candidate['property_type'] else 0.0
            
            # Year built similarity
            if subject['year_built'] and candidate['year_built']:
                year_diff = abs(subject['year_built'] - candidate['year_built'])
                similarities['year_built'] = 1 / (1 + year_diff/50)  # Normalize by 50 years
            else:
                similarities['year_built'] = 0
            
            # Lot size similarity
            if subject['lot_size'] and candidate['lot_size']:
                lot_diff = abs(subject['lot_size'] - candidate['lot_size'])
                similarities['lot_size'] = 1 / (1 + lot_diff/max(subject['lot_size'], candidate['lot_size']))
            else:
                similarities['lot_size'] = 0
            
            # Calculate weighted average
            weighted_similarity = sum(
                similarities[feature] * weight 
                for feature, weight in self.feature_weights.items()
            )
            
            return weighted_similarity
        except Exception as e:
            logger.error(f"Error calculating property similarity: {str(e)}")
            return 0.0
    
    def find_comparable_properties(self, subject: Dict, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
        """Find top N comparable properties for a subject property."""
        try:
            # Calculate similarities for all candidates
            similarities = []
            for candidate in candidates:
                if candidate['id'] != subject['id']:  # Exclude subject property
                    similarity = self.calculate_property_similarity(subject, candidate)
                    similarities.append((candidate, similarity))
            
            # Sort by similarity and get top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_comps = similarities[:top_n]
            
            # Add similarity scores to results
            results = []
            for comp, similarity in top_comps:
                comp['similarity_score'] = similarity
                results.append(comp)
            
            return results
        except Exception as e:
            logger.error(f"Error finding comparable properties: {str(e)}")
            return []

    def _is_address_match(self, addr1, addr2):
        if not addr1 or not addr2:
            return False
        addr1 = self._normalize_address(addr1)
        addr2 = self._normalize_address(addr2)
        if addr1 == addr2:
            return True
        if addr1 in addr2 or addr2 in addr1:
            return True
        components1 = set(addr1.split())
        components2 = set(addr2.split())
        common_components = components1.intersection(components2)
        if len(common_components) >= 2:
            for comp in common_components:
                if comp.isdigit() or len(comp) > 3:
                    return True
        similarity = 1 - (distance(addr1, addr2) / max(len(addr1), len(addr2)))
        return similarity >= 0.6 