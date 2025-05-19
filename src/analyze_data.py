from data.data_processor import DataProcessor
import json

def main():
    # Initialize the data processor
    processor = DataProcessor("appraisals_dataset.json")
    
    # Analyze the dataset
    print("Analyzing dataset...")
    stats = processor.analyze_dataset()
    
    # Print the statistics in a readable format
    print("\nDataset Statistics:")
    print(f"Total Appraisals: {stats['total_appraisals']}")
    print(f"Average Candidates per Appraisal: {stats['avg_candidates_per_appraisal']:.2f}")
    print(f"Average Comps per Appraisal: {stats['avg_comps_per_appraisal']:.2f}")
    
    print("\nProperty Types:")
    for prop_type, count in stats['property_types'].items():
        print(f"- {prop_type}: {count}")
    
    print("\nFeatures:")
    for feature, info in stats['features'].items():
        print(f"- {feature}:")
        print(f"  Type: {info['type']}")
        print(f"  Unique Values: {info['unique_values']}")

if __name__ == "__main__":
    main() 