import requests
import json
from typing import Optional, Dict, Any

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


def extract_part_details(api_response: Dict[str, Any]) -> list:
    """
    Extract specific fields from the Mouser API response for each part.
    
    Args:
        api_response (Dict[str, Any]): The raw API response from search_by_part_number
        
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
                'ROHSStatus': part.get('ROHSStatus', '')
            }
            parts_details.append(part_details)
    
    return parts_details


# result = search_by_part_number("LHA-03-TS")
# if result:
#     # Process the response data
#     # print(result)
#     part_details = extract_part_details(result)
#     # print(part_details)