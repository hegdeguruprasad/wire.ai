"""
    This fuction file is used to Download datasheets from URLs stored in MongoDB and save them locally.

"""


import requests
from pymongo import MongoClient
from datetime import datetime
import logging
from urllib.parse import urlparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("datasheet_download.log"),
        logging.StreamHandler()  # This will print logs to console as well
    ]
)
logger = logging.getLogger(__name__)

def download_datasheets(db_url, db_name, collection_name, download_dir):
    """
    Download datasheets from URLs stored in MongoDB and save them locally.

    Args:
        db_url (str): MongoDB connection URI
        db_name (str): Name of the database
        collection_name (str): Name of the collection
        download_dir (str): Directory to store downloaded datasheets
    """
    try:
        # Create download directory if it doesn't exist
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Download directory created/verified at: {download_path.absolute()}")

        # Connect to MongoDB
        logger.info("Attempting to connect to MongoDB...")
        client = MongoClient(db_url)
        db= client[db_name]
        collection = db[collection_name]
        logger.info("Successfully connected to MongoDB")

        # Query documents with datasheet URLs
        logger.info("Querying documents with datasheet URLs...")
        doc_count = collection.count_documents({"DataSheetUrl": {"$exists": True, "$ne": ""}})
        logger.info(f"Found {doc_count} documents with DataSheetUrl")

        cursor = collection.find({"DataSheetUrl": {"$exists": True, "$ne": ""}})
        
        download_count = 0
        error_count = 0

        for doc in cursor:
            try:
                url = doc.get("DataSheetUrl")
                part_number = doc.get("Part_Number", "unknown")
                
                logger.info(f"Processing part number: {part_number}")
                logger.info(f"Datasheet URL: {url}")

                if not url:
                    logger.warning(f"Skipping {part_number} - No URL found")
                    continue

                # Generate filename
                file_extension = Path(urlparse(url).path).suffix or ".pdf"
                filename = f"{part_number}{file_extension}"
                filepath = download_path / filename
                
                if filepath.exists():
                    logger.info(f"Datasheet already exists for {part_number}")
                    continue

                logger.info(f"Downloading datasheet for {part_number}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                }
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                
                # Save the file
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                download_count += 1
                logger.info(f"Successfully downloaded datasheet for {part_number}")
                
                # Update MongoDB with local file path
                collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "LocalDatasheetPath": str(filepath),
                            "DatasheetDownloadDate": datetime.now()
                        }
                    }
                )
                logger.info(f"Updated MongoDB record for {part_number}")
                
            except requests.exceptions.RequestException as e:
                error_count += 1
                logger.error(f"Failed to download datasheet for {part_number}: {str(e)}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {part_number}: {str(e)}")
        
        logger.info(f"Download process completed")
        logger.info(f"Total documents processed: {doc_count}")
        logger.info(f"Successfully downloaded: {download_count}")
        logger.info(f"Errors encountered: {error_count}")
                
    except Exception as e:
        logger.error(f"Fatal error in download_datasheets: {str(e)}", exc_info=True)
    finally:
        client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    # Example usage
    download_datasheets(
        db_url = "mongodb://localhost:27017",
        db_name="Data",
        collection_name="MouserComponents",
        download_dir="Datasheets"
    )
