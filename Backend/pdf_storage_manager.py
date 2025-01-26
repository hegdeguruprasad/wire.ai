import os
import sys
import logging
import pymongo
import gridfs
import PyPDF2
from typing import Dict, Any, Optional
from pymongo import MongoClient
from datetime import datetime

class PDFStorageManager:
    def __init__(self):
        """
        Initialize MongoDB connection and GridFS for PDF storage
        Uses connection parameters from db_connection.py
        """
        try:
            # Add project root to Python path
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            sys.path.insert(0, project_root)

            # Import the existing get_mongodb_client function specifically from Data Extraction
            data_extraction_path = os.path.join(project_root, 'Data Extraction')
            sys.path.insert(0, data_extraction_path)

            # Direct import from Data Extraction.db_connection
            from db_connection import get_mongodb_client, DATABASE

            # Use the existing connection method
            self.client = get_mongodb_client()
            
            if not self.client:
                raise ValueError("Failed to establish MongoDB connection")
            
            # Use the database name from db_connection
            self.db = self.client[DATABASE]
            self.fs = gridfs.GridFS(self.db)
            self.pdf_collection = self.db['pdf_metadata']
            
            # Configure logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("MongoDB connection established successfully")
        
        except ImportError:
            logging.error("Could not import get_mongodb_client from Data Extraction.db_connection")
            raise
        except Exception as e:
            logging.error(f"Error initializing MongoDB connection: {e}")
            raise

    def extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF using PyPDF2
        
        Args:
            file_path (str): Path to the PDF file
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {e}")
            return ''

    def store_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a PDF file in MongoDB using GridFS
        
        Args:
            file_path (str): Path to the PDF file
            metadata (dict, optional): Additional metadata for the PDF
        
        Returns:
            str: GridFS file ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Extract text from PDF
        pdf_text = self.extract_pdf_text(file_path)

        # Prepare metadata
        default_metadata = {
            'filename': os.path.basename(file_path),
            'uploaded_at': datetime.utcnow(),
            'file_size': os.path.getsize(file_path),
            'content_type': 'application/pdf'
        }
        
        # Merge default and user-provided metadata
        if metadata:
            default_metadata.update(metadata)

        # Store PDF file in GridFS
        with open(file_path, 'rb') as pdf_file:
            file_id = self.fs.put(
                pdf_file, 
                filename=os.path.basename(file_path), 
                **default_metadata
            )

        # Store extracted text and metadata in separate collection
        text_metadata = {
            'file_id': file_id,
            'filename': default_metadata['filename'],
            'extracted_text': pdf_text,
            **default_metadata
        }
        
        self.pdf_collection.insert_one(text_metadata)
        
        self.logger.info(f"PDF stored successfully: {file_id}")
        return str(file_id)

    def retrieve_pdf(self, file_id: str) -> Dict[str, Any]:
        """
        Retrieve PDF file and its metadata
        
        Args:
            file_id (str): GridFS file ID
        
        Returns:
            dict: PDF file and metadata
        """
        try:
            # Retrieve file from GridFS
            pdf_file = self.fs.get(file_id)
            
            # Retrieve metadata from PDF metadata collection
            metadata = self.pdf_collection.find_one({'file_id': file_id})
            
            return {
                'file': pdf_file,
                'metadata': metadata
            }
        except Exception as e:
            self.logger.error(f"Error retrieving PDF: {e}")
            return None

    def search_pdfs(self, query: Dict[str, Any]) -> list:
        """
        Search PDFs based on metadata
        
        Args:
            query (dict): Search criteria
        
        Returns:
            list: Matching PDF metadata
        """
        return list(self.pdf_collection.find(query))

    def delete_pdf(self, file_id: str):
        """
        Delete a PDF file from GridFS and metadata collection
        
        Args:
            file_id (str): GridFS file ID
        """
        try:
            # Delete file from GridFS
            self.fs.delete(file_id)
            
            # Delete metadata
            self.pdf_collection.delete_one({'file_id': file_id})
            
            self.logger.info(f"PDF deleted successfully: {file_id}")
        except Exception as e:
            self.logger.error(f"Error deleting PDF: {e}")