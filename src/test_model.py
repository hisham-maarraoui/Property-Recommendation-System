import json
import logging
from models.recommender import PropertyRecommender
from evaluation.evaluator import RecommendationEvaluator
import re
from data.processor import PropertyDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(file_path: str):
    """Load test data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('appraisals', [])
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return []

def clean_gla(gla_str):
    if not gla_str:
        return 0.0
    try:
        # Remove commas, non-numeric, and non-dot characters
        gla_str = re.sub(r'[^0-9.]', '', gla_str.replace(',', ''))
        return float(gla_str) if gla_str else 0.0
    except:
        return 0.0

def clean_price(price_str):
    if not price_str:
        return 0.0
    try:
        price_str = re.sub(r'[^0-9.]', '', price_str.replace(',', ''))
        return float(price_str) if price_str else 0.0
    except:
        return 0.0

def extract_bedrooms(bed_str):
    if not bed_str:
        return 0
    match = re.search(r'\d+', str(bed_str))
    return int(match.group()) if match else 0

def prepare_test_cases(appraisals, properties):
    """Prepare test cases from appraisals and properties data.
    The comps in each appraisal are the ground truth, and the properties list is the candidate pool for recommendations.
    """
    processor = PropertyDataProcessor()
    
    # Preprocess all properties first, filter out None results
    processed_properties = [p for p in (processor.preprocess_property(prop) for prop in properties) if p is not None]
    
    test_cases = []
    for i, appraisal in enumerate(appraisals):
        subject = appraisal.get('subject', {})
        comps = appraisal.get('comps', [])
        
        # Process subject
        processed_subject = processor.preprocess_property(subject)
        
        # Log raw comp data for first few test cases
        if i < 3:
            logger.info(f"\nTest Case {i + 1} - Raw Comp Data:")
            for j, comp in enumerate(comps):
                logger.info(f"Comp {j + 1}:")
                logger.info(f"  Address: {comp.get('address', '')} {comp.get('city_province', '')}")
                logger.info(f"  Price: {comp.get('sale_price', '')}")
                logger.info(f"  GLA: {comp.get('gla', '')}")
                logger.info(f"  Bedrooms: {comp.get('bed_count', '')}")
                logger.info(f"  Property Type: {comp.get('property_type', '')}")
                logger.info(f"  Structure Type: {comp.get('structure_type', '')}")
                logger.info(f"  Description: {comp.get('description', '')}")
                logger.info(f"  Public Remarks: {comp.get('public_remarks', '')}")
        
        # Process comps as ground truth
        processed_comps = []
        for comp in comps:
            # Try to match comp with existing property
            comp_address = f"{comp.get('address', '')} {comp.get('city_province', '')}"
            matched_property = None
            
            for prop in processed_properties:
                if processor._is_address_match(comp_address, prop['address']):
                    matched_property = prop
                    break
            
            if matched_property:
                processed_comps.append(matched_property)
            else:
                # Process comp directly if no match found
                processed_comp = processor.preprocess_property(comp)
                if processed_comp is not None:
                    processed_comps.append(processed_comp)
        
        # Log processed comp data for first few test cases
        if i < 3:
            logger.info(f"\nTest Case {i + 1} - Processed Comp Data:")
            for j, comp in enumerate(processed_comps):
                logger.info(f"Comp {j + 1}:")
                logger.info(f"  Address: {comp.get('address', '')}")
                logger.info(f"  Price: {comp.get('price', '')}")
                logger.info(f"  GLA: {comp.get('gla', '')}")
                logger.info(f"  Bedrooms: {comp.get('bedrooms', '')}")
                logger.info(f"  Property Type: {comp.get('property_type', '')}")
                logger.info(f"  Structure Type: {comp.get('structure_type', '')}")
        
        # Use the processed properties as the candidate pool for recommendations
        test_cases.append((processed_subject, processed_comps, processed_properties))
    
    return test_cases

def main():
    # Initialize recommender
    recommender = PropertyRecommender()
    
    # Load test data
    test_data = load_test_data('appraisals_dataset.json')
    test_cases = prepare_test_cases(test_data, test_data)
    
    if not test_cases:
        logger.error("No test cases found")
        return
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(recommender)
    
    # Evaluate each test case
    all_metrics = []
    for i, (subject, actual_comps, properties) in enumerate(test_cases):
        # Load properties for this test case
        recommender.properties = properties
        
        # Get recommendations
        recommendations = recommender.get_recommendations(subject)
        
        # Evaluate recommendations
        metrics = evaluator.evaluate_recommendations(subject, recommendations, actual_comps)
        if metrics:
            all_metrics.append(metrics)
            
            # Print results for this case
            logger.info(f"\nTest Case {i+1}:")
            logger.info(f"Subject: {subject.get('address', 'N/A')}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
            logger.info(f"NDCG: {metrics['ndcg']:.3f}")
            logger.info(f"MRR: {metrics['mrr']:.3f}")
            logger.info(f"Price Error: {metrics['price_error']:.3f}")
            
            # Print recommendations
            logger.info("\nTop 3 Recommendations:")
            for j, rec in enumerate(recommendations, 1):
                logger.info(f"\n{j}. {rec.get('address', 'N/A')}")
                logger.info(f"   Price: ${rec.get('price', 0):,.2f}")
                logger.info(f"   Similarity Score: {rec.get('similarity_score', 0):.3f}")
                logger.info(f"   Explanation: {rec.get('explanation', 'N/A')}")
            
            # Print actual comps for comparison
            logger.info("\nActual Comparable Properties:")
            for j, comp in enumerate(actual_comps, 1):
                logger.info(f"\n{j}. {comp.get('address', 'N/A')}")
                logger.info(f"   Price: ${comp.get('close_price', 0):,.2f}")
                logger.info(f"   GLA: {comp.get('gla', 0)} sq ft")
                logger.info(f"   Bedrooms: {comp.get('bedrooms', 0)}")
    
    # Print average metrics
    if all_metrics:
        avg_metrics = {
            'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
            'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
            'f1_score': sum(m['f1_score'] for m in all_metrics) / len(all_metrics),
            'ndcg': sum(m['ndcg'] for m in all_metrics) / len(all_metrics),
            'mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics),
            'price_error': sum(m['price_error'] for m in all_metrics) / len(all_metrics)
        }
        
        logger.info("\nOverall Average Metrics:")
        logger.info(f"Precision: {avg_metrics['precision']:.3f}")
        logger.info(f"Recall: {avg_metrics['recall']:.3f}")
        logger.info(f"F1 Score: {avg_metrics['f1_score']:.3f}")
        logger.info(f"NDCG: {avg_metrics['ndcg']:.3f}")
        logger.info(f"MRR: {avg_metrics['mrr']:.3f}")
        logger.info(f"Average Price Error: {avg_metrics['price_error']:.3f}")

if __name__ == "__main__":
    main() 