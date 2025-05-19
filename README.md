# Property Recommendation System

A machine learning-based system for recommending comparable properties for real estate valuation decisions.

## Features

- Property similarity calculation based on multiple features
- Location-based filtering using Haversine distance
- Price range consideration
- Property type matching
- Explainable recommendations with detailed reasoning
- Comprehensive evaluation metrics

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── processor.py      # Data preprocessing and feature extraction
│   ├── models/
│   │   └── recommender.py    # Property recommendation model
│   ├── evaluation/
│   │   └── evaluator.py      # Model evaluation metrics
│   └── test_model.py         # Test script
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your property data in JSON format with the following structure:

```json
{
  "appraisals": [
    {
      "subject": {
        "id": "property_id",
        "address": "property_address",
        "price": price_value,
        "gla": living_area,
        "bedrooms": number_of_bedrooms,
        "latitude": latitude_value,
        "longitude": longitude_value,
        ...
      },
      "comparables": [
        {
          "id": "comparable_id",
          ...
        }
      ]
    }
  ]
}
```

2. Run the test script:

```bash
python src/test_model.py
```

## Evaluation Metrics

The system evaluates recommendations using the following metrics:

- Precision: Ratio of correctly recommended properties
- Recall: Ratio of actual comparable properties found
- F1 Score: Harmonic mean of precision and recall
- NDCG: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- Price Error: Average price difference between recommendations and subject property

## Model Features

The recommendation system considers the following features when finding comparable properties:

- Gross Living Area (GLA)
- Number of bedrooms
- Property price
- Location (latitude/longitude)
- Property type
- Structure type
- Style
- Close date

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
