import json
from typing import Dict, List
from ..data.data_processor import DataProcessor
from ..models.baseline_similarity import rank_candidates, extract_features

def evaluate_baseline(data_path: str) -> Dict:
    """Evaluate the baseline similarity model on the dataset."""
    processor = DataProcessor(data_path)
    processor.load_data()
    total_appraisals = len(processor.data)
    correct_recommendations = 0
    total_recommendations = 0

    # Debug: Print features for the first appraisal
    if processor.data:
        first_appraisal = processor.data[0]
        subject = first_appraisal.get('subject', {})
        candidates = first_appraisal.get('properties', [])
        labeled_comps = first_appraisal.get('comps', [])
        print("First Appraisal Subject Features:")
        print(extract_features(subject))
        print("First Appraisal Labeled Comps Features:")
        for i, comp in enumerate(labeled_comps):
            print(f"Labeled Comp {i+1}:", extract_features(comp))
        print("First Appraisal Candidate Features (first 3):")
        for i, cand in enumerate(candidates[:3]):
            print(f"Candidate {i+1}:", extract_features(cand))

    for i, appraisal in enumerate(processor.data):
        if i < 3:  # Print for the first 3 appraisals
            print(f"Raw subject property data for appraisal {i+1}:")
            print(appraisal['subject'])
        subject = appraisal.get('subject', {})
        candidates = appraisal.get('properties', [])
        labeled_comps = appraisal.get('comps', [])
        recommended_comps = rank_candidates(subject, candidates, top_k=3)
        # Count how many recommended comps are in the labeled comps
        for rec in recommended_comps:
            if rec in labeled_comps:
                correct_recommendations += 1
        total_recommendations += len(recommended_comps)

    accuracy = correct_recommendations / total_recommendations if total_recommendations > 0 else 0
    return {
        'total_appraisals': total_appraisals,
        'correct_recommendations': correct_recommendations,
        'total_recommendations': total_recommendations,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    results = evaluate_baseline("appraisals_dataset.json")
    print("Evaluation Results:")
    print(json.dumps(results, indent=2)) 