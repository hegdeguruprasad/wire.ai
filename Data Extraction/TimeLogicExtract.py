"""
This code is used to get the Data from Mouser with Time Logic. The mouser API has a limit of 
    50 results returned per call 
    Up to 30 calls per minute 
    Up to 1,000 calls per day
The code will use different API keys to make API calls and will rotate them to avoid getting blocked.

"""


import time
import logging
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
import json
from StoreData import store_mouser_data, store_failed_part_number
from mouserAPI import extract_part_details,search_by_part_number_list
import requests
from collections import deque
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = deque(api_keys)
        self.call_counts = {key: {'daily': 0, 'minute_calls': [], 'last_used': None} for key in api_keys}
        self.load_state()

    def save_state(self):
        state = {
            'call_counts': {
                k: {
                    'daily': v['daily'],
                    'minute_calls': [t.timestamp() for t in v['minute_calls']],
                    'last_used': v['last_used'].timestamp() if v['last_used'] else None
                }
                for k, v in self.call_counts.items()
            }
        }
        with open('api_state.json', 'w') as f:
            json.dump(state, f)

    def load_state(self):
        try:
            with open('api_state.json', 'r') as f:
                state = json.load(f)
                for key, data in state['call_counts'].items():
                    if key in self.call_counts:
                        self.call_counts[key]['daily'] = data['daily']
                        self.call_counts[key]['minute_calls'] = [
                            datetime.fromtimestamp(t) for t in data['minute_calls']
                        ]
                        self.call_counts[key]['last_used'] = (
                            datetime.fromtimestamp(data['last_used']) 
                            if data['last_used'] else None
                        )
        except FileNotFoundError:
            pass

    def get_next_available_key(self) -> str:
        now = datetime.now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        for _ in range(len(self.api_keys)):
            current_key = self.api_keys[0]
            key_data = self.call_counts[current_key]

            # Reset daily count if it's a new day
            if key_data['last_used'] and key_data['last_used'].date() < now.date():
                key_data['daily'] = 0
                key_data['minute_calls'] = []

            # Remove calls older than 1 minute
            key_data['minute_calls'] = [
                t for t in key_data['minute_calls']
                if (now - t).total_seconds() < 60
            ]

            if (key_data['daily'] < 1000 and 
                len(key_data['minute_calls']) < 30):
                return current_key

            # Rotate to next key
            self.api_keys.rotate(-1)

        return None

    def record_api_call(self, api_key: str):
        now = datetime.now()
        self.call_counts[api_key]['daily'] += 1
        self.call_counts[api_key]['minute_calls'].append(now)
        self.call_counts[api_key]['last_used'] = now
        self.save_state()

def extract_mouser_data_with_rate_limit(
    part_numbers: Set[str],
    api_keys: List[str],
    batch_size: int = 50,
    db_url: str = "mongodb://localhost:27017"
) -> Dict[str, int]:
    """
    Extract data from Mouser API with rate limiting and multiple API keys.
    
    Args:
        part_numbers: Set of part numbers to process
        api_keys: List of Mouser API keys to use
        batch_size: Number of parts to process in each API call
        db_url: MongoDB connection URL
        
    Returns:
        Dict with statistics about the processing
    """
    stats = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'remaining': len(part_numbers)
    }

    key_manager = APIKeyManager(api_keys)
    part_numbers = list(part_numbers)
    
    try:
        while part_numbers:
            # Get next available API key
            api_key = key_manager.get_next_available_key()
            if not api_key:
                logger.info("All API keys have reached their limits. Waiting for 60 seconds...")
                time.sleep(60)
                continue

            # Process next batch
            batch = part_numbers[:batch_size]
            
            try:
                # Make API call
                response = search_by_part_number_list(batch, api_key)
                key_manager.record_api_call(api_key)
                
                if response and 'SearchResults' in response:
                    # Extract specific fields using the existing function and match with original part numbers
                    parts_details = []
                    matched_parts = set() # Keep track of successfully matched part numbers

                    for part in response['SearchResults'].get('Parts', []):
                        # Find the matching original part number
                        original_part = None
                        for batch_part in batch:
                            if batch_part.lower() in part.get('ManufacturerPartNumber', '').lower():
                                original_part = batch_part
                                break
                        if original_part:
                            parts_details.extend(extract_part_details({'SearchResults': {'Parts': [part]}}, original_part))
                            matched_parts.add(original_part)  # Add to matched parts

                    if parts_details:
                        # Store successful results
                        store_mouser_data(parts_details, db_url)
                        stats['successful'] += len(parts_details)

                    
                    else:
                        # Store failed parts (those in the batch but not matched in the response)
                        failed_parts = [part for part in batch if part not in matched_parts]
                        for part in failed_parts:
                            store_failed_part_number(
                                part, 
                                "No part details found in response", 
                                db_url
                            )
                        stats['failed'] += len(failed_parts)
                else:
                    # Store failed parts
                    for part in batch:
                        store_failed_part_number(
                            part, 
                            "Invalid API response", 
                            db_url
                        )
                    stats['failed'] += len(batch)

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Store failed parts
                for part in batch:
                    store_failed_part_number(part, str(e), db_url)
                stats['failed'] += len(batch)

            # Update progress
            stats['processed'] += len(batch)
            stats['remaining'] -= len(batch)
            part_numbers = part_numbers[batch_size:]
            
            logger.info(f"Progress: {stats['processed']}/{stats['processed'] + stats['remaining']} "
                       f"(Success: {stats['successful']}, Failed: {stats['failed']})")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        return stats
