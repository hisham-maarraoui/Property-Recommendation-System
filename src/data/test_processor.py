import json
from processor import PropertyDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processor():
    # Create processor instance
    processor = PropertyDataProcessor()
    
    # Sample data from our dataset
    sample_data = {
        "orderID": "4762739",
        "subject": {
            "address": "7180 207 HWY Halifax NS B0J2L0",
            "subject_city_province_zip": "West Chezzetcook, NS B0J2L0",
            "effective_date": "Apr/17/2025",
            "municipality_district": "Halifax Regional Municipality - West Chezzetcook",
            "site_dimensions": "See Schedule A and or Plot Map",
            "lot_size_sf": "72745+/-SqFt",
            "units_sq_ft": "SqFt",
            "year_built": "2011",
            "structure_type": "Detached",
            "roofing": "Asphalt Shingle",
            "effective_age": "10",
            "style": "1.5 Storey",
            "construction": "Log",
            "remaining_economic_life": "50",
            "windows": "Wood",
            "basement": "Full/Finished",
            "exterior_finish": "Log",
            "basement_area": "1060",
            "foundation_walls": "Poured Concrete",
            "flooring": "Carpet",
            "plumbing_lines": "Copper, PEX, ABS",
            "heating": "Radiant",
            "fuel_type": "Electric",
            "water_heater": "80 +/- gl Electric",
            "cooling": "Ductless mini split",
            "room_count": "6",
            "num_beds": "3",
            "room_total": "6",
            "main_lvl_area": "1060",
            "second_lvl_area": "440",
            "third_lvl_area": "",
            "gla": "1500 SqFt",
            "subject_age": "14+/-yrs",
            "num_baths": "2:1",
            "condition": "Average"
        },
        "comps": [
            {
                "distance_to_subject": "3.73 KM",
                "prop_type": "Detached",
                "stories": "1 Storey",
                "address": "64 Deermist Dr",
                "city_province": "Porters Lake NS B3E 1P3",
                "sale_date": "Jan/16/2025",
                "sale_price": "800,000",
                "dom": "141+/-",
                "location_similarity": "Inferior",
                "lot_size": "80212+/-SqFt",
                "age": "11+/-",
                "condition": "Similar",
                "gla": "1602+/-SqFt",
                "room_count": "6",
                "bed_count": "3",
                "bath_count": "2:0",
                "basement_finish": "Full/Finished",
                "parking": "Dbl. Att. Gar.",
                "neighborhood": ""
            }
        ]
    }
    
    # Process subject property
    logger.info("Processing subject property...")
    subject = processor.preprocess_property(sample_data["subject"])
    if subject:
        logger.info("Subject property processed successfully:")
        logger.info(f"ID: {subject['id']}")
        logger.info(f"Address: {subject['address']}")
        logger.info(f"Bedrooms: {subject['bedrooms']}")
        logger.info(f"Bathrooms: {subject['bathrooms']}")
        logger.info(f"GLA: {subject['gla']}")
        logger.info(f"Lot Size: {subject['lot_size']}")
        logger.info(f"Year Built: {subject['year_built']}")
        logger.info(f"Property Type: {subject['property_type']}")
    else:
        logger.error("Failed to process subject property")
    
    # Process comp property
    logger.info("\nProcessing comp property...")
    comp = processor.preprocess_property(sample_data["comps"][0])
    if comp:
        logger.info("Comp property processed successfully:")
        logger.info(f"ID: {comp['id']}")
        logger.info(f"Address: {comp['address']}")
        logger.info(f"Bedrooms: {comp['bedrooms']}")
        logger.info(f"Bathrooms: {comp['bathrooms']}")
        logger.info(f"GLA: {comp['gla']}")
        logger.info(f"Lot Size: {comp['lot_size']}")
        logger.info(f"Price: {comp['price']}")
        logger.info(f"Property Type: {comp['property_type']}")
    else:
        logger.error("Failed to process comp property")
    
    # Test property similarity
    if subject and comp:
        logger.info("\nTesting property similarity...")
        similarity = processor.calculate_property_similarity(subject, comp)
        logger.info(f"Similarity score: {similarity:.4f}")

if __name__ == "__main__":
    test_processor() 