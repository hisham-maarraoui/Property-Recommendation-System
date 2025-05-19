import unittest
import json
import os
from datetime import datetime
from src.data.processor import PropertyDataProcessor
from src.models.recommender import PropertyRecommender
from src.explanation.explainer import PropertyExplainer
from src.models.feedback import FeedbackManager

class TestPropertySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and initialize components."""
        # Create sample property data
        cls.sample_properties = [
            {
                "id": "1",
                "address": "123 Main St",
                "city": "Toronto",
                "province": "ON",
                "postal_code": "M5V 2H1",
                "latitude": 43.6532,
                "longitude": -79.3832,
                "bedrooms": 3,
                "bathrooms": 2,
                "gla": 2000,
                "lot_size": 5000,
                "price": 750000,
                "year_built": 2010,
                "property_type": "single_family",
                "style": "modern",
                "close_date": "2023-01-15"
            },
            {
                "id": "2",
                "address": "456 Oak Ave",
                "city": "Toronto",
                "province": "ON",
                "postal_code": "M5V 2H2",
                "latitude": 43.6533,
                "longitude": -79.3833,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "gla": 2100,
                "lot_size": 4800,
                "price": 780000,
                "year_built": 2012,
                "property_type": "single_family",
                "style": "modern",
                "close_date": "2023-02-01"
            },
            {
                "id": "3",
                "address": "789 Pine Rd",
                "city": "Toronto",
                "province": "ON",
                "postal_code": "M5V 2H3",
                "latitude": 43.6534,
                "longitude": -79.3834,
                "bedrooms": 4,
                "bathrooms": 3,
                "gla": 2500,
                "lot_size": 6000,
                "price": 950000,
                "year_built": 2015,
                "property_type": "single_family",
                "style": "contemporary",
                "close_date": "2023-03-01"
            }
        ]
        
        # Save sample data to temporary file
        cls.test_data_file = "test_properties.json"
        with open(cls.test_data_file, 'w') as f:
            json.dump({"appraisals": cls.sample_properties}, f)
        
        # Initialize components
        cls.processor = PropertyDataProcessor()
        cls.recommender = PropertyRecommender()
        cls.explainer = PropertyExplainer()
        cls.feedback_manager = FeedbackManager("test_feedback.json")

        # Preprocess properties to get generated IDs
        cls.processed_properties = [cls.processor.preprocess_property(p) for p in cls.sample_properties]
        cls.id_map = {i: p['id'] for i, p in enumerate(cls.processed_properties)}
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.test_data_file):
            os.remove(cls.test_data_file)
        if os.path.exists("test_feedback.json"):
            os.remove("test_feedback.json")
    
    def test_data_processing(self):
        """Test property data preprocessing."""
        # Test property preprocessing
        processed = self.processor.preprocess_property(self.sample_properties[0])
        self.assertIsNotNone(processed)
        self.assertEqual(processed['bedrooms'], 3)
        self.assertEqual(processed['gla'], 2000)
        self.assertEqual(processed['price'], 750000)
        
        # Test address normalization
        normalized = self.processor._normalize_address("123 Main St, Unit 4")
        self.assertEqual(normalized, "123 main street")
        
        # Test numeric feature extraction
        gla = self.processor._extract_numeric_feature(
            self.sample_properties[0],
            ['gla', 'size', 'square_feet']
        )
        self.assertEqual(gla, 2000)
    
    def test_property_similarity(self):
        """Test property similarity calculations."""
        # Test location similarity
        loc_sim = self.processor.calculate_location_similarity(
            self.sample_properties[0],
            self.sample_properties[1]
        )
        self.assertGreater(loc_sim, 0)
        self.assertLessEqual(loc_sim, 1)
        
        # Test overall similarity
        sim = self.processor.calculate_property_similarity(
            self.sample_properties[0],
            self.sample_properties[1]
        )
        self.assertGreater(sim, 0)
        self.assertLessEqual(sim, 1)
    
    def test_recommendations(self):
        """Test property recommendations."""
        # Load properties
        self.recommender.load_properties(self.test_data_file)
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.sample_properties[0]
        )
        
        # Verify recommendations
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 3)
        if recommendations:
            self.assertIn('similarity_score', recommendations[0])
            self.assertIn('explanation', recommendations[0])
            self.assertIn('explanation_confidence', recommendations[0])
    
    def test_explainer(self):
        """Test property explainer."""
        # Train explainer
        self.explainer.train(self.sample_properties)
        
        # Generate explanation
        explanation = self.explainer.explain_comparison(
            self.sample_properties[0],
            self.sample_properties[1]
        )
        
        # Verify explanation
        self.assertIsNotNone(explanation)
        self.assertIn('explanations', explanation)
        self.assertIn('similarity_score', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('feature_importance', explanation)
    
    def test_feedback_system(self):
        """Test feedback system."""
        # Add feedback using generated IDs
        subject_id = self.id_map[0]
        comp_id = self.id_map[1]
        success = self.feedback_manager.add_feedback(
            subject_id, comp_id, 4, "Good match"
        )
        self.assertTrue(success)
        
        # Get feedback stats
        stats = self.feedback_manager.get_feedback_stats()
        self.assertEqual(stats['total_entries'], 1)
        self.assertEqual(stats['average_rating'], 4.0)
        
        # Update model with numeric features
        success = self.feedback_manager.update_model(self.processed_properties)
        self.assertTrue(success)
        
        # Get weight history
        history = self.feedback_manager.get_weight_history()
        self.assertGreater(len(history), 0)
    
    def test_end_to_end(self):
        """Test end-to-end recommendation flow with feedback."""
        # Load properties
        self.recommender.load_properties(self.test_data_file)
        
        # Get initial recommendations
        initial_recs = self.recommender.get_recommendations(
            self.sample_properties[0]
        )
        
        # Add feedback for a different property using generated IDs
        subject_id = self.id_map[0]
        comp_id = self.id_map[2]
        self.recommender.add_feedback(
            subject_id,
            comp_id,
            5,
            "Excellent match"
        )
        
        # Get updated recommendations
        updated_recs = self.recommender.get_recommendations(
            self.sample_properties[0]
        )
        
        # Verify recommendations changed
        self.assertNotEqual(
            initial_recs[0]['id'],
            updated_recs[0]['id']
        )
        
        # Verify feedback stats
        stats = self.recommender.get_feedback_stats()
        self.assertEqual(stats['total_entries'], 1)
        self.assertEqual(stats['average_rating'], 5.0)
    
    def test_real_dataset(self):
        """Test the system with the real appraisals dataset."""
        # Load real dataset
        real_data_file = "appraisals_dataset.json"
        if not os.path.exists(real_data_file):
            self.skipTest("Real dataset file not found")
        
        # Initialize new recommender for real data
        real_recommender = PropertyRecommender()
        success = real_recommender.load_properties(real_data_file)
        self.assertTrue(success)
        
        # Load the dataset to get a sample property
        with open(real_data_file, 'r') as f:
            real_data = json.load(f)
            sample_property = real_data['appraisals'][0]
        
        # Get recommendations
        recommendations = real_recommender.get_recommendations(sample_property)
        
        # Verify recommendations
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 3)
        if recommendations:
            self.assertIn('similarity_score', recommendations[0])
            self.assertIn('explanation', recommendations[0])
            self.assertIn('explanation_confidence', recommendations[0])
            
            # Verify recommendation quality
            for rec in recommendations:
                self.assertGreater(rec['similarity_score'], 0.5)  # High similarity threshold
                self.assertGreater(rec['explanation_confidence'], 0.7)  # High confidence threshold
                
                # Verify explanation components
                self.assertIsInstance(rec['explanation'], list)
                self.assertGreater(len(rec['explanation']), 0)
                
                # Verify feature importance
                self.assertIn('feature_importance', rec)
                self.assertIsInstance(rec['feature_importance'], dict)
                self.assertGreater(len(rec['feature_importance']), 0)

if __name__ == '__main__':
    unittest.main() 