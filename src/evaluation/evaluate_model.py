import json
import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.property_model import PropertyModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model: PropertyModel):
        """Initialize the model evaluator."""
        self.model = model
        self.metrics = {}
    
    def _calculate_price_error(self, predicted_price: float, actual_price: float) -> float:
        """Calculate the percentage error in price prediction."""
        if actual_price <= 0:
            return float('inf')
        return abs(predicted_price - actual_price) / actual_price * 100
    
    def _extract_property_id(self, prop: Dict[str, Any]) -> str:
        """Extract property ID from a property dictionary."""
        try:
            if isinstance(prop, dict):
                # Try to get ID from various possible locations
                if 'id' in prop:
                    return str(prop['id'])
                elif 'property_id' in prop:
                    return str(prop['property_id'])
                elif 'property' in prop and isinstance(prop['property'], dict):
                    if 'id' in prop['property']:
                        return str(prop['property']['id'])
                    elif 'property_id' in prop['property']:
                        return str(prop['property']['property_id'])
                
                # If no ID found, try to generate one from address and price
                address = str(prop.get('address', '')).strip()
                price = str(prop.get('price', prop.get('sale_price', prop.get('close_price', '0'))))
                if address and price:
                    unique_str = f"{address}_{price}"
                    return str(hash(unique_str) % 1000000)  # Generate a 6-digit numeric ID
            
            return None
        except Exception as e:
            logger.error(f"Error extracting property ID: {str(e)}")
            return None
    
    def _calculate_similarity_metrics(self, recommended: List[Dict[str, Any]], 
                                   actual_comps: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for similarity-based evaluation."""
        try:
            # Extract property IDs
            recommended_ids = set()
            for r in recommended:
                prop_id = self._extract_property_id(r)
                if prop_id:
                    recommended_ids.add(prop_id)
            
            actual_ids = set()
            for comp in actual_comps:
                prop_id = self._extract_property_id(comp)
                if prop_id:
                    actual_ids.add(prop_id)
            
            if not recommended_ids or not actual_ids:
                return {'precision': 0, 'recall': 0, 'f1_score': 0}
            
            true_positives = len(recommended_ids.intersection(actual_ids))
            precision = true_positives / len(recommended_ids)
            recall = true_positives / len(actual_ids)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {'precision': precision, 'recall': recall, 'f1_score': f1}
        except Exception as e:
            logger.error(f"Error calculating similarity metrics: {str(e)}")
            return {'precision': 0, 'recall': 0, 'f1_score': 0}
    
    def _calculate_ranking_metrics(self, ranked_properties: List[Dict[str, Any]], 
                                 actual_comps: List[Dict[str, Any]]) -> Dict[str, float]:
        try:
            # Extract property IDs from actual comparables
            actual_ids = set()
            for comp in actual_comps:
                prop_id = self._extract_property_id(comp)
                if prop_id:
                    actual_ids.add(prop_id)
            
            if not actual_ids:
                return {'ndcg': 0, 'mrr': 0}
            
            # Calculate DCG
            dcg = 0
            for i, prop in enumerate(ranked_properties):
                prop_id = self._extract_property_id(prop)
                if prop_id in actual_ids:
                    dcg += 1 / np.log2(i + 2)
            
            # Calculate IDCG (ideal DCG)
            idcg = 0
            for i in range(min(len(actual_ids), len(ranked_properties))):
                idcg += 1 / np.log2(i + 2)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # Calculate MRR
            mrr = 0
            for i, prop in enumerate(ranked_properties):
                prop_id = self._extract_property_id(prop)
                if prop_id in actual_ids:
                    mrr = 1 / (i + 1)
                    break
            
            return {'ndcg': ndcg, 'mrr': mrr}
            
        except Exception as e:
            logger.error(f"Error calculating ranking metrics: {str(e)}")
            return {'ndcg': 0, 'mrr': 0}
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        try:
            total_price_error = 0
            total_similarity_metrics = {'precision': 0, 'recall': 0, 'f1_score': 0}
            total_ranking_metrics = {'ndcg': 0, 'mrr': 0}
            valid_cases = 0
            
            for case_idx, case in enumerate(test_data):
                try:
                    target_property = case['target_property']
                    actual_comps = case['comparable_properties']
                    neighborhood_properties = case['neighborhood_properties']
                    
                    # Ensure target property has an ID
                    if 'id' not in target_property:
                        target_property['id'] = f"target_{valid_cases}"
                    
                    # Train the model on the neighborhood properties for this case
                    self.model = PropertyModel()  # Re-initialize for each case
                    self.model.train(neighborhood_properties)
                    
                    # Get similar properties
                    similar_props = self.model.find_similar_properties(target_property['id'])
                    
                    if not similar_props:
                        logger.warning(f"No similar properties found for target {target_property['id']}")
                        continue
                    
                    # Debug: Print recommended and actual comparable IDs for the first valid case
                    if valid_cases == 0:
                        recommended_ids = [self._extract_property_id(r) for r in similar_props]
                        actual_ids = [self._extract_property_id(c) for c in actual_comps]
                        logger.info(f"[DEBUG] Test case {case_idx}: Recommended IDs: {recommended_ids}")
                        logger.info(f"[DEBUG] Test case {case_idx}: Actual Comp IDs: {actual_ids}")
                    
                    # Calculate similarity metrics
                    sim_metrics = self._calculate_similarity_metrics(similar_props, actual_comps)
                    for metric, value in sim_metrics.items():
                        total_similarity_metrics[metric] += value
                    
                    # Calculate ranking metrics
                    ranked_props = self.model.rank_properties(similar_props, target_property)
                    rank_metrics = self._calculate_ranking_metrics(ranked_props, actual_comps)
                    for metric, value in rank_metrics.items():
                        total_ranking_metrics[metric] += value
                    
                    # Calculate price error using weighted average based on similarity scores
                    if similar_props:
                        prices = []
                        weights = []
                        for prop in similar_props:
                            if isinstance(prop, dict):
                                price = None
                                if 'price' in prop:
                                    price = prop['price']
                                elif 'property' in prop and isinstance(prop['property'], dict):
                                    price = prop['property'].get('price')
                                
                                if price and price > 0:
                                    prices.append(price)
                                    weights.append(prop.get('similarity_score', 1.0))
                        
                        if prices and target_property.get('price', 0) > 0:
                            weighted_price = np.average(prices, weights=weights)
                            price_error = self._calculate_price_error(weighted_price, target_property['price'])
                            if price_error != float('inf'):
                                total_price_error += price_error
                                valid_cases += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating case: {str(e)}")
                    continue
            
            # Calculate averages
            if valid_cases > 0:
                return {
                    'average_price_error': total_price_error / valid_cases,
                    'average_precision': total_similarity_metrics['precision'] / valid_cases,
                    'average_recall': total_similarity_metrics['recall'] / valid_cases,
                    'average_f1_score': total_similarity_metrics['f1_score'] / valid_cases,
                    'average_ndcg': total_ranking_metrics['ndcg'] / valid_cases,
                    'average_mrr': total_ranking_metrics['mrr'] / valid_cases
                }
            else:
                return {
                    'average_price_error': 0,
                    'average_precision': 0,
                    'average_recall': 0,
                    'average_f1_score': 0,
                    'average_ndcg': 0,
                    'average_mrr': 0
                }
                
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return {
                'average_price_error': 0,
                'average_precision': 0,
                'average_recall': 0,
                'average_f1_score': 0,
                'average_ndcg': 0,
                'average_mrr': 0
            }

def main():
    try:
        with open('data/processed/test_data.json', 'r') as f:
            test_data = json.load(f)
        model = PropertyModel()  # Dummy instance for evaluator
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(test_data)
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        with open('data/evaluation/results.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logger.error(f"Error in main evaluation: {str(e)}")

if __name__ == "__main__":
    main() 