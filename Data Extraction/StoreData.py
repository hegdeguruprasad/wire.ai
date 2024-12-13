from pymongo import MongoClient
from typing import List, Dict, Any, Set
import logging
from mouserAPI import search_by_part_number, extract_part_details
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_url = "mongodb://localhost:27017"


def get_all_part_numbers(db_url):
    """
    Extract all part numbers from the Components collection.
    
    Args:
        db_url (str): MongoDB connection URL
        
    Returns:
        Set[str]: Set of unique part numbers
    """
    try:
        client = MongoClient(db_url)
        db = client["Data"]
        components_collection = db["ComponentsCopy"]

        
        part_numbers = set()
        
        # Iterate through all documents in Components collection
        cursor = components_collection.find({})
        for doc in cursor:
            for subcategory in doc.get('subcategories', []):
                for component in subcategory.get('extracted_components', []):
                    part_number = component.get('part_number')
                    if part_number:
                        part_numbers.add(part_number)
        
        logger.info(f"Found {len(part_numbers)} unique part numbers")
        client.close()
        return part_numbers
        
    except Exception as e:
        logger.error(f"Error getting part numbers: {str(e)}")
        return set()

def store_mouser_data(data: List[Dict[str, Any]], db_url):
    """
    Store Mouser API data in MouserComponents collection.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing component data from Mouser API
        db_url (str): MongoDB connection URL
        
    Returns:
        bool: True if data was stored successfully, False otherwise
    """
    try:
        client = MongoClient(db_url)
        db = client["Data"]
        collection = db["MouserComponents"]

        
        # Insert or update documents
        for component in data:
            # Using ManufacturerPartNumber as a unique identifier
            query = {"ManufacturerPartNumber": component["ManufacturerPartNumber"]}
            update = {
                "$set": {
                    **component,
                    "last_updated": datetime.utcnow()
                }
            }
            collection.update_one(query, update, upsert=True)
            
        logger.info(f"Successfully stored {len(data)} components in MouserComponents")
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing data in MongoDB: {str(e)}")
        return False

def store_failed_part_number(part_number: str, reason: str, db_url):
    """
    Store failed part numbers in FailedComponents collection.
    
    Args:
        part_number (str): The part number that failed
        reason (str): Reason for failure
        db_url (str): MongoDB connection URL
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        client = MongoClient(db_url)
        db = client["Data"]
        Failed_collection = db["FailedComponents"]

        
        document = {
            "part_number": part_number,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }
        
        Failed_collection.insert_one(document)
        logger.info(f"Stored failed part number: {part_number}")
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing failed part number: {str(e)}")
        return False

def process_all_components(db_url: str = "mongodb://localhost:27017/") -> Dict[str, int]:
    """
    Process all components from Components collection and store results in MouserComponents
    and FailedComponents collections.
    
    Args:
        db_url (str): MongoDB connection URL
        
    Returns:
        Dict[str, int]: Statistics about the processing
    """
    stats = {
        "total_parts": 0,
        "successful": 0,
        "failed": 0
    }
    
    # Get all part numbers
    part_numbers = get_all_part_numbers(db_url)
    stats["total_parts"] = len(part_numbers)
    
    # Process each part number
    for part_number in part_numbers:
        # Search for the component using Mouser API
        api_response = search_by_part_number(part_number)
        
        if not api_response:
            store_failed_part_number(part_number, "No API response", db_url)
            stats["failed"] += 1
            continue
        
        # Extract relevant details from the API response
        part_details = extract_part_details(api_response)
        
        if not part_details:
            store_failed_part_number(part_number, "No part details found", db_url)
            stats["failed"] += 1
            continue
        
        # Store the data in MouserComponents collection
        if store_mouser_data(part_details, db_url):
            stats["successful"] += 1
        else:
            store_failed_part_number(part_number, "Failed to store in database", db_url)
            stats["failed"] += 1
    
    logger.info(f"Processing completed. Stats: {stats}")
    return stats


# if __name__ == "__main__":
#     stats = process_all_components()
#     print(stats)
