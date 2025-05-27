from typing import Dict, List, Tuple
import logging
from src.models.recommender import PropertyRecommender
import numpy as np
from datetime import datetime
import re
from Levenshtein import distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    def __init__(self, recommender: PropertyRecommender):
        self.recommender = recommender
        self.metrics = {}
    
    def evaluate_recommendations(self, subject, recommendations, actual_comps):
        """Evaluate recommendations against actual comparable properties."""
        if not recommendations:
            logging.warning("No recommendations to evaluate")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ndcg': 0.0,
                'mrr': 0.0,
                'price_error': 0.0
            }

        # Extract property IDs or addresses for matching
        recommended_ids = []
        actual_ids = []
        
        # Process recommendations
        for rec in recommendations:
            if isinstance(rec, dict):
                # Try to get ID first
                prop_id = rec.get('id')
                if not prop_id:
                    # Fall back to address
                    address = rec.get('address', '')
                    if address:
                        # Clean and normalize address
                        address = self._normalize_address(address)
                        recommended_ids.append(address)
            else:
                logging.warning(f"Unexpected recommendation format: {type(rec)}")
                continue

        # Process actual comparables
        for comp in actual_comps:
            if isinstance(comp, dict):
                # Try to get ID first
                comp_id = comp.get('id')
                if not comp_id:
                    # Fall back to address
                    address = comp.get('address', '')
                    if address:
                        # Clean and normalize address
                        address = self._normalize_address(address)
                        actual_ids.append(address)
            else:
                logging.warning(f"Unexpected comparable format: {type(comp)}")
                continue

        # Calculate metrics
        try:
            # Find matches using flexible matching
            matches = []
            for rec_id in recommended_ids:
                for actual_id in actual_ids:
                    if self._is_address_match(rec_id, actual_id):
                        matches.append(rec_id)
                        break

            # Calculate metrics
            precision = self._calculate_precision(len(matches), len(recommended_ids))
            recall = self._calculate_recall(len(matches), len(actual_ids))
            f1_score = self._calculate_f1_score(precision, recall)
            ndcg = self._calculate_ndcg(recommendations, actual_comps)
            mrr = self._calculate_mrr(recommendations, actual_comps)
            price_error = self._calculate_price_error(recommendations, subject)

            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'ndcg': ndcg,
                'mrr': mrr,
                'price_error': price_error
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ndcg': 0.0,
                'mrr': 0.0,
                'price_error': 0.0
            }

    def _normalize_address(self, address):
        """Normalize address for comparison."""
        if not address:
            return ""
        # Convert to lowercase
        address = address.lower()
        # Remove unit/suite numbers (e.g., 'unit 101 -', 'apt 2,', etc.)
        address = re.sub(r'\b(unit|apt|suite|#)\s*\d+[\s\-:,]*', '', address)
        # Standardize common abbreviations
        abbreviations = {
            ' st ': ' street ',
            ' ave ': ' avenue ',
            ' rd ': ' road ',
            ' dr ': ' drive ',
            ' blvd ': ' boulevard ',
            ' cres ': ' crescent ',
            ' crt ': ' court ',
            ' cir ': ' circle ',
            ' pl ': ' place ',
            ' ln ': ' lane ',
            ' sq ': ' square ',
            ' hwy ': ' highway ',
            ' ter ': ' terrace ',
            ' pky ': ' parkway ',
            ' ct ': ' court ',
            ' n/a ': ' ',
        }
        # Pad with spaces for whole word replacement
        address = f' {address} '
        for abbr, full in abbreviations.items():
            address = address.replace(abbr, full)
        address = address.strip()
        # Remove punctuation
        address = re.sub(r'[^\w\s]', '', address)
        # Remove extra whitespace
        address = ' '.join(address.split())
        return address

    def _is_address_match(self, addr1, addr2):
        """Check if two addresses match using flexible criteria."""
        if not addr1 or not addr2:
            return False
        
        # Normalize addresses
        addr1 = self._normalize_address(addr1)
        addr2 = self._normalize_address(addr2)
        
        # Check for exact match after normalization
        if addr1 == addr2:
            return True
        
        # Check for partial match (one address contains the other)
        if addr1 in addr2 or addr2 in addr1:
            return True
        
        # Split addresses into components
        components1 = set(addr1.split())
        components2 = set(addr2.split())
        
        # Check for significant overlap in components
        common_components = components1.intersection(components2)
        if len(common_components) >= 2:  # Require at least 2 common words
            # Check if the common components include important parts (street number, name, or type)
            important_components = [c for c in common_components if c.isdigit() or len(c) > 3]
            if len(important_components) >= 1:  # Require at least 1 important component
                return True
        
        # Check for similarity using Levenshtein distance
        similarity = 1 - (distance(addr1, addr2) / max(len(addr1), len(addr2)))
        if similarity >= 0.5:  # Lower threshold to 50%
            return True
        
        # Check for street name match
        street1 = ' '.join([c for c in addr1.split() if not c.isdigit()])
        street2 = ' '.join([c for c in addr2.split() if not c.isdigit()])
        if street1 and street2:
            street_similarity = 1 - (distance(street1, street2) / max(len(street1), len(street2)))
            if street_similarity >= 0.7:  # 70% similarity for street names
                return True
        
        return False

    def _calculate_precision(self, matches: int, total_recommended: int) -> float:
        """Calculate precision of recommendations."""
        if total_recommended == 0:
            return 0.0
        return matches / total_recommended

    def _calculate_recall(self, matches: int, total_actual: int) -> float:
        """Calculate recall of recommendations."""
        if total_actual == 0:
            return 0.0
        return matches / total_actual

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_ndcg(self, recommendations: List[Dict], actual_comps: List[Dict]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not recommendations or not actual_comps:
            return 0.0

        # Create a relevance mapping for actual comparables
        relevance_map = {}
        for comp in actual_comps:
            if isinstance(comp, dict):
                address = self._normalize_address(comp.get('address', ''))
                if address:
                    relevance_map[address] = 1.0

        # Calculate DCG
        dcg = 0.0
        for i, rec in enumerate(recommendations):
            if isinstance(rec, dict):
                address = self._normalize_address(rec.get('address', ''))
                if address in relevance_map:
                    dcg += relevance_map[address] / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (ideal case where all actual comparables are ranked first)
        idcg = 0.0
        for i in range(min(len(relevance_map), len(recommendations))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, recommendations: List[Dict], actual_comps: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not recommendations or not actual_comps:
            return 0.0

        # Create a set of actual comparable addresses
        actual_addresses = {
            self._normalize_address(comp.get('address', ''))
            for comp in actual_comps
            if isinstance(comp, dict) and comp.get('address')
        }

        # Find the rank of the first match
        for i, rec in enumerate(recommendations):
            if isinstance(rec, dict):
                address = self._normalize_address(rec.get('address', ''))
                if address in actual_addresses:
                    return 1.0 / (i + 1)

        return 0.0

    def _calculate_price_error(self, recommendations: List[Dict], subject: Dict) -> float:
        """Calculate average price error of recommendations."""
        if not recommendations or not subject:
            return 0.0

        subject_price = float(subject.get('price', 0))
        if subject_price == 0:
            return 0.0

        total_error = 0.0
        valid_recommendations = 0

        for rec in recommendations:
            if isinstance(rec, dict):
                rec_price = float(rec.get('price', 0))
                if rec_price > 0:
                    error = abs(rec_price - subject_price) / subject_price
                    total_error += error
                    valid_recommendations += 1

        return total_error / valid_recommendations if valid_recommendations > 0 else 0.0

    def evaluate_dataset(self, test_cases: List[Tuple[Dict, List[Dict]]]) -> Dict:
        """Evaluate the recommendation system on a dataset of test cases."""
        try:
            all_metrics = []
            for subject, actual_comps in test_cases:
                metrics = self.evaluate_recommendations(subject, actual_comps)
                if metrics:
                    all_metrics.append(metrics)
            
            if not all_metrics:
                return {}
            
            # Calculate average metrics
            avg_metrics = {
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
                'ndcg': np.mean([m['ndcg'] for m in all_metrics]),
                'mrr': np.mean([m['mrr'] for m in all_metrics]),
                'price_error': np.mean([m['price_error'] for m in all_metrics])
            }
            
            return avg_metrics
        except Exception as e:
            logger.error(f"Error evaluating dataset: {str(e)}")
            return {} 