import requests
import json
from typing import Optional, Dict, Any, List

MOUSER_API_KEY = "80052764-786d-4f9e-bca6-97e7e7f0f009"
# MOUSER_API_KEY = "121351ad-b2db-4d5c-9028-ecf8f12a4902"
MOUSER_BASE_URL = "https://api.mouser.com/api/v2"

def search_by_part_number(part_number: str) -> Optional[Dict[str, Any]]:
    """
    Search for a component using its part number in Mouser's database.
    
    Args:
        part_number (str): The part number to search for
        
    Returns:
        Optional[Dict[str, Any]]: The API response data if successful, None if failed
    """
    endpoint = f"{MOUSER_BASE_URL}/search/partnumber"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "SearchByPartRequest": {
            "mouserPartNumber": part_number,
            "partSearchOptions": ""
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            params={"apiKey": MOUSER_API_KEY},
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
        
    except requests.RequestException as e:
        print(f"Error searching for part {part_number}: {str(e)}")
        return None

# The key difference is that search_by_part_number is designed to handle only one part number at a time,
# while make_mouser_api_call is designed to handle multiple part numbers in a single API call 
# using Mouser's batch capability (using the | separator)

def search_by_part_number_list(part_numbers: List[str], api_key: str) -> Dict[str, Any]:
    """Make batch API call to Mouser"""
    endpoint = "https://api.mouser.com/api/v2/search/partnumber"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "SearchByPartRequest": {
            "mouserPartNumber": "|".join(part_numbers),
            "partSearchOptions": ""
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            params={"apiKey": api_key},
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
        
    except requests.RequestException as e:
        logger.error(f"API call failed: {str(e)}")
        raise

def extract_part_details(api_response: Dict[str, Any], original_part_number: str = None) -> list:
    """
    Extract specific fields from the Mouser API response for each part.
    
    Args:
        api_response (Dict[str, Any]): The raw API response from search_by_part_number
        original_part_number (str, optional): The original part number used for the search
        
    Returns:
        list: List of dictionaries containing the specified fields for each part
    """
    parts_details = []
    
    if api_response and 'SearchResults' in api_response and 'Parts' in api_response['SearchResults']:
        for part in api_response['SearchResults']['Parts']:
            part_details = {
                'DataSheetUrl': part.get('DataSheetUrl', ''),
                'Description': part.get('Description', ''),
                'ImagePath': part.get('ImagePath', ''),
                'Category': part.get('Category', ''),
                'Manufacturer': part.get('Manufacturer', ''),
                'ManufacturerPartNumber': part.get('ManufacturerPartNumber', ''),
                'MouserPartNumber': part.get('MouserPartNumber', ''),
                'ProductDetailUrl': part.get('ProductDetailUrl', ''),
                'ROHSStatus': part.get('ROHSStatus', ''),
                'Part_Number': original_part_number
            }
            parts_details.append(part_details)
    
    return parts_details


result = search_by_part_number("TSM4ZJ101KR10")
if result:
    # Process the response data
    # print(result)
    part_details = extract_part_details(result,"TSM4ZJ101KR10")
    print(part_details)