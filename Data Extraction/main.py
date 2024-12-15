from TimeLogicExtract import extract_mouser_data_with_rate_limit
from StoreData import get_all_part_numbers

# List of your API keys
api_keys = [
    "80052764-786d-4f9e-bca6-97e7e7f0f009",
    "121351ad-b2db-4d5c-9028-ecf8f12a4902",
    "92acb6ff-4504-4d9e-afb8-9ccf4c176cf9",
    "75d8a244-0a43-48c1-98ad-8924a4b67adb"
    # Add more API keys here
]

def main():
    # Get part numbers from database
    part_numbers = get_all_part_numbers("mongodb://localhost:27017")
    
    # Start extraction with rate limiting
    stats = extract_mouser_data_with_rate_limit(
        part_numbers=part_numbers,
        api_keys=api_keys,
        batch_size=50  # Maximum allowed by Mouser
    )
    
    print("Extraction completed!")
    print(f"Total processed: {stats['processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")

if __name__ == "__main__":
    main()