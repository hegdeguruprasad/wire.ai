# server.py

"""
RAG System API Server
--------------------
This is the main server file that provides API endpoints for the RAG system.

To run this server:
1. Ensure all requirements are installed:
   pip install fastapi uvicorn python-multipart

2. Save this file as 'server.py'

3. Run the server:
   uvicorn server:app --reload

The server will start at http://localhost:8000
API documentation will be available at http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import shutil
import os
from datetime import datetime
from bson import ObjectId
import logging
from pymongo import MongoClient
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
import requests
from uuid import uuid4
import logging
import time
from dotenv import load_dotenv
from bson import ObjectId  
import json  
from langchain_core.documents import Document

load_dotenv()



# Import your existing RAG system
from epsilonRAG import (
    DocumentProcessor,
    DocumentStore,
    VectorStore,
    RAGSystem,
    setup_api_keys
)


# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for processing and querying documents using RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize storage systems

doc_store = DocumentStore(
    # connection_string="mongodb://localhost:27017",
    connection_string = "mongodb+srv://ga3hegde:51uDnJ3iB0cwOdSB@rag.y0614.mongodb.net/?retryWrites=true&w=majority&appName=RAG",
    database="VectorDatabase",
    docs= "processed_documents",
    sections= "document_sections",
    embeddings= "embeddings_tracking"

)
# VectorStore Initialization using LocalHost
# vector_store = VectorStore(
#     dbname="VectorDB",
#     user="postgres",
#     password="12345",
#     host="localhost",
#     port="5432"
# )


# VectorStore Initialization using Supabase
vector_store = VectorStore(
    # db_url="postgresql://postgres.qcqeeulzbgezcrkvrhoh:WireAi#12345$@aws-0-us-west-1.pooler.supabase.com:6543/postgres",
    db_url= "postgresql://postgres.qcqeeulzbgezcrkvrhoh:9CkzjF0rXdgmF3bV@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
    collection_name="pdf_vectors"
)


# Setup API keys
setup_api_keys()

# Pydantic models
class QuestionRequest(BaseModel):
    doc_id: str
    question: str

class ProcessingStatus(BaseModel):
    doc_id: str
    status: str
    timestamp: datetime

class DocumentResponse(BaseModel):
    doc_id: str
    file_name: str
    processed_date: datetime
    section_count: int

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    previous_messages: Optional[List[Dict[str, str]]] = None





class ComponentStore:
    def __init__(self, connection_string: str, database: str, collection: str):
        logging.info(f"Initializing ComponentStore with database: {database}, collection: {collection}")
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.components = self.db[collection]

    def search_component(self, query: str) -> Optional[dict]:
        """Search for a component by exact ManufacturerPartNumber."""
        try:
            logging.info(f"Searching for component with ManufacturerPartNumber: {query}")
            start_time = time.time()
            
            # Exact match query
            component = self.components.find_one({"ManufacturerPartNumber": query})
            
            elapsed_time = time.time() - start_time
            logging.info(f"Component search completed in {elapsed_time:.2f} seconds")
            
            if component:
                logging.info("Component found")
                # Convert ObjectId to string
                component["id"] = str(component.pop("_id"))
                return component
                
            logging.info("No component found")
            return None
            
        except Exception as e:
            logging.error(f"Error searching component: {str(e)}", exc_info=True)
            return None




# Component model definition
class Component(BaseModel):
    id: str
    ManufacturerPartNumber: str
    Category: str
    DataSheetUrl: str
    Description: str
    ImagePath: str
    Manufacturer: str
    MouserPartNumber: str
    Part_Number: str
    ProductDetailUrl: str
    ROHSStatus: str
    last_updated: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

component_store = ComponentStore(
    # connection_string="mongodb://localhost:27017",
    connection_string = "mongodb+srv://ga3hegde:51uDnJ3iB0cwOdSB@rag.y0614.mongodb.net/?retryWrites=true&w=majority&appName=RAG",
    database="Data",
    collection="MouserComponents"
)
# Global dictionary to store processing status
processing_status = {}
# Add this right after your processing_status dictionary declaration


# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Helper Functions
def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to disk and return file path."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path






async def process_document_background(file_path: str, doc_id: str):
    """Process document in background and update status."""
    try:
        # Process the document
        processor = DocumentProcessor(file_path)
        processed_sections, chunked_sections = processor.process_document()
        
        # Store document sections
        doc_store.store_document(file_path, processed_sections, chunked_sections)
        
        # Setup RAG system and store vectors
        rag_system = RAGSystem(doc_store, vector_store)
        rag_system.setup_retriever(processed_sections, chunked_sections, doc_id)
        
        # Update section count in MongoDB
        doc_store.docs.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "section_count": len(processed_sections)
                }
            }
        )
        
        # Update in-memory status to completed
        processing_status[doc_id] = {
            "status": "completed",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Update in-memory status on failure
        processing_status[doc_id] = {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now()
        }
        raise





async def download_and_process_datasheet(
    component_id: str,
    datasheet_url: str
) -> str:
    """Downloads and processes a PDF datasheet."""
    try:
        logging.info(f"Starting download for component {component_id}")
        
        # Check if we've already processed this component's datasheet
        existing_doc = doc_store.docs.find_one({
            "component_id": component_id,
            "status": "completed"
        })
        
        if existing_doc:
            logging.info(f"Found existing completed document: {existing_doc['_id']}")
            return str(existing_doc["_id"])

        # Find or create document entry
        doc = doc_store.docs.find_one({
            "component_id": component_id,
            "status": {"$nin": ["completed", "failed"]}
        })
        
        if not doc:
            logging.info("Creating new document entry")
            doc_metadata = {
                "file_name": f"{component_id}_datasheet.pdf",
                "processed_date": datetime.now(),
                "section_count": 0,
                "component_id": component_id,
                "status": "downloading"
            }
            result = doc_store.docs.insert_one(doc_metadata)
            doc_id = str(result.inserted_id)
        else:
            doc_id = str(doc["_id"])
            
        logging.info(f"Using document ID: {doc_id}")

        # Download PDF
        logging.info(f"Downloading PDF from {datasheet_url}")
        # response = requests.get(datasheet_url)
        response = requests.get(datasheet_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        # Save PDF temporarily
        filename = f"{component_id}_datasheet.pdf"
        file_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        logging.info(f"PDF saved to {file_path}")

        # Process the PDF directly
        await process_pdf_in_background(file_path, doc_id, component_id)
        
        return doc_id
        
    except Exception as e:
        logging.error(f"Error in download_and_process_datasheet: {str(e)}", exc_info=True)
        if doc_id:
            doc_store.docs.update_one(
                {"_id": ObjectId(doc_id)},
                {
                    "$set": {
                        "status": "failed",
                        "error": str(e),
                        "error_at": datetime.now()
                    }
                }
            )
        raise
    

async def process_datasheet_background(file_path: str, doc_id: str):
    """Process datasheet in background and update status."""
    try:
        # Process the document
        processor = DocumentProcessor(file_path)
        processed_sections, chunked_sections = processor.process_document()
        
        # Store document sections
        doc_store.store_document(file_path, processed_sections, chunked_sections)
        
        # Setup RAG system and store vectors
        rag_system = RAGSystem(doc_store, vector_store)
        rag_system.setup_retriever(processed_sections, chunked_sections, doc_id)
        
        # Update document metadata
        doc_store.docs.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "section_count": len(processed_sections)
                }
            }
        )
        
        # Update processing status
        processing_status[doc_id] = {
            "status": "completed",
            "timestamp": datetime.now()
        }
        
        # Clean up temporary file
        os.remove(file_path)
        
    except Exception as e:
        # Update status on failure
        processing_status[doc_id] = {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now()
        }
        # Clean up temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise






# Update document creation to include status sync
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    try:
        file_path = save_uploaded_file(file)
        
        # Check if document exists
        existing_doc = doc_store.docs.find_one({"file_name": file.filename})
        if existing_doc:
            doc_id = str(existing_doc["_id"])
            # Sync status for existing document
            sync_document_status(doc_id)
            return DocumentResponse(
                doc_id=doc_id,
                file_name=existing_doc["file_name"],
                processed_date=existing_doc["processed_date"],
                section_count=existing_doc.get("section_count", 0)
            )
        
        # Create new document
        doc_metadata = {
            "file_name": file.filename,
            "processed_date": datetime.now(),
            "status": "pending"
        }
        result = doc_store.docs.insert_one(doc_metadata)
        doc_id = str(result.inserted_id)
        
        # Initialize status after MongoDB insertion
        processing_status[doc_id] = {
            "status": "pending",
            "timestamp": doc_metadata["processed_date"]
        }
        
        background_tasks.add_task(
            process_document_background,
            file_path,
            doc_id
        )
        
        return DocumentResponse(
            doc_id=doc_id,
            file_name=file.filename,
            processed_date=doc_metadata["processed_date"],
            section_count=0
        )
        
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(500, f"Error processing document: {str(e)}")



# Helper function for status synchronization
def sync_document_status(doc_id: str) -> None:
    """Synchronize in-memory status with MongoDB document status."""
    try:
        doc = doc_store.docs.find_one({"_id": ObjectId(doc_id)})
        if doc:
            current_time = datetime.now()
            processing_status[doc_id] = {
                "status": doc["status"],
                "timestamp": doc.get("completed_at") or doc.get("error_at") or doc.get("processed_date") or current_time
            }
            if doc.get("error"):
                processing_status[doc_id]["error"] = doc["error"]
    except Exception as e:
        logging.error(f"Error syncing status for doc_id {doc_id}: {str(e)}")







@app.get("/documents/{doc_id}/status", response_model=ProcessingStatus)
async def get_processing_status(doc_id: str):
    """Get the processing status of a document with synchronized checks."""
    try:
        # First check MongoDB for the latest status
        doc = doc_store.docs.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            raise HTTPException(404, "Document not found")
        
        # Update in-memory status from MongoDB if needed
        if doc_id not in processing_status or \
           processing_status[doc_id]["status"] != doc["status"]:
            processing_status[doc_id] = {
                "status": doc["status"],
                "timestamp": doc.get("completed_at") or doc.get("error_at") or doc["processed_date"]
            }
            if doc.get("error"):
                processing_status[doc_id]["error"] = doc["error"]
        
        return ProcessingStatus(
            doc_id=doc_id,
            status=processing_status[doc_id]["status"],
            timestamp=processing_status[doc_id]["timestamp"]
        )
        
    except Exception as e:
        logging.error(f"Error getting status for doc_id {doc_id}: {str(e)}")
        raise HTTPException(500, f"Error fetching status: {str(e)}")


# Modify the query endpoint
@app.post("/documents/{doc_id}/query", response_model=Dict)
async def query_document(doc_id: str, request: QuestionRequest):
    """Query a processed document with a question."""
    # Check if document exists
    doc = doc_store.docs.find_one({"_id": ObjectId(doc_id)})
    if not doc:
        raise HTTPException(404, "Document not found")
        
    # Check processing status
    status = processing_status.get(doc_id, {}).get("status")
    if status != "completed":
        raise HTTPException(400, "Document processing not completed")
    
    try:
        # Get processed sections
        processed_sections, chunked_sections = doc_store.get_document_sections(doc_id)
        if not processed_sections or not chunked_sections:
            raise HTTPException(500, "Document sections not found")
        
        # Setup RAG system
        rag_system = RAGSystem(doc_store, vector_store)
        retriever = rag_system.setup_retriever(processed_sections, chunked_sections, doc_id)
        
        # Process question
        response = rag_system.process_question(retriever, request.question)
        
        return {
            "doc_id": doc_id,
            "question": request.question,
            "answer": response
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing query: {str(e)}")

@app.get("/documents/", response_model=List[DocumentResponse])
async def list_documents():
    """
    List all processed documents.
    
    Returns:
    - List of documents with their metadata
    """
    try:
        documents = []
        for doc in doc_store.docs.find():
            documents.append(
                DocumentResponse(
                    doc_id=str(doc["_id"]),
                    file_name=doc["file_name"],
                    processed_date=doc["processed_date"],
                    section_count=doc.get("section_count", 0)
                )
            )
        return documents
        
    except Exception as e:
        raise HTTPException(500, f"Error fetching documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and its associated data.
    """
    try:
        # Delete document metadata
        result = doc_store.docs.delete_one({"_id": doc_id})
        if result.deleted_count == 0:
            raise HTTPException(404, "Document not found")
        
        # Delete sections
        doc_store.sections.delete_many({"document_id": doc_id})
        
        # Delete embeddings tracking
        doc_store.embeddings.delete_many({"document_id": doc_id})
        
        # Remove from processing status
        if doc_id in processing_status:
            del processing_status[doc_id]
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        raise HTTPException(500, f"Error deleting document: {str(e)}")
    





@app.get("/components/search/", response_model=Optional[Component])
async def search_component(query: str, background_tasks: BackgroundTasks):
    """Search for a component and initiate datasheet processing."""
    try:
        # Step 1: Find the component
        component = component_store.search_component(query)
        if not component:
            raise HTTPException(404, "Component not found")
            
        # Step 2: Find existing processed document
        existing_doc = doc_store.docs.find_one({
            "component_id": component['id'],
            "status": "completed"  # Only get completed documents
        })
        
        if existing_doc:
            # Step 3: Add doc_id to component response
            component['doc_id'] = str(existing_doc["_id"])
            logging.info(f"Found existing document with ID: {component['doc_id']}")
        elif component.get('DataSheetUrl'):
            # Step 4: Create new document entry if none exists
            doc_metadata = {
                "file_name": f"{component['id']}_datasheet.pdf",
                "processed_date": datetime.now(),
                "section_count": 0,
                "component_id": component['id'],
                "status": "processing"  # Set initial status
            }
            result = doc_store.docs.insert_one(doc_metadata)
            component['doc_id'] = str(result.inserted_id)
            
            # Step 5: Start background processing
            background_tasks.add_task(
                download_and_process_datasheet,
                component['id'],
                component['DataSheetUrl']
            )
        
        return component
        
    except Exception as e:
        logging.error(f"Error in search endpoint: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error searching component: {str(e)}")    
    



@app.post("/documents/{doc_id}/chat", response_model=Dict)
async def chat_with_document(doc_id: str, request: ChatRequest):
    """Chat with a processed document with conversation history."""
    try:
        # Get processed sections
        processed_sections, chunked_sections = doc_store.get_document_sections(doc_id)
        if not processed_sections or not chunked_sections:
            raise HTTPException(500, "Document sections not found")
        
        # Setup RAG system
        rag_system = RAGSystem(doc_store, vector_store)
        retriever = rag_system.setup_retriever(processed_sections, chunked_sections, doc_id)
        
        # Build conversation context
        conversation_context = ""
        if request.previous_messages:
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in request.previous_messages[-3:]  # Last 3 messages for context
            ])
        
        # Process question using RAG with context
        contextualized_question = f"""
        Previous conversation:
        {conversation_context}
        
        Current question: {request.question}
        """
        
        answer = rag_system.process_question(retriever, contextualized_question)
        
        return {
            "answer": answer,
            "doc_id": doc_id,
            "conversation_id": request.conversation_id or str(uuid4())
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing chat: {str(e)}")





async def process_pdf_in_background(file_path: str, doc_id: str, component_id: str):
    """Process PDF in background with improved data handling."""
    try:
        logging.info(f"Starting PDF processing for doc_id: {doc_id}")
        
        # Process document
        processor = DocumentProcessor(file_path)
        logging.info("Created DocumentProcessor, starting document processing...")
        processed_sections, chunked_sections = processor.process_document()
        
        # Log section count and structure
        section_count = len(processed_sections)
        logging.info(f"Document processed with {section_count} sections")

        # Convert BSON types to JSON-safe formats
        processed_sections = json.loads(
            json.dumps(processed_sections, default=str)
        )
        chunked_sections = json.loads(
            json.dumps(chunked_sections, default=str)
        )

        # Store document sections with existing doc_id
        logging.info("Storing document sections...")
        doc_store.store_document(file_path, processed_sections, chunked_sections, doc_id)
        
        # Set up RAG system and store vectors
        logging.info("Setting up RAG system and storing vectors...")
        rag_system = RAGSystem(doc_store, vector_store)
        
        # Create and store vectors for each section
        for result in processed_sections:
            element_id = result["element_id"]
            if not doc_store.has_embeddings(doc_id, element_id):
                section_doc = Document(
                    page_content=result["title_text_summary"],
                    metadata={
                        "category": "Section",
                        "element_id": element_id,
                        "doc_id": doc_id
                    }
                )
                vector_store.store.add_documents([section_doc])
                doc_store.track_embedding(doc_id, element_id, "Section")

            # Store vectors for content items
            for summary_content in result["contents"]:
                content_id = summary_content["unique_id"]
                if not doc_store.has_embeddings(doc_id, content_id):
                    content_doc = Document(
                        page_content=summary_content["summary"],
                        metadata={
                            "category": summary_content["category"],
                            "element_id": content_id,
                            "parent_section": element_id,
                            "doc_id": doc_id
                        }
                    )
                    vector_store.store.add_documents([content_doc])
                    doc_store.track_embedding(doc_id, content_id, summary_content["category"])

        # Update status to complete
        doc_store.docs.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "status": "completed",
                    "section_count": section_count,
                    "completed_at": datetime.now().isoformat()
                }
            }
        )
        logging.info(f"Processing completed for doc_id: {doc_id}")
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logging.error(error_msg, exc_info=True)
        
        # Update status to failed
        doc_store.docs.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "status": "failed",
                    "error": error_msg,
                    "error_at": datetime.now().isoformat()
                }
            }
        )
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logging.error(f"Error cleaning up file {file_path}: {str(e)}")

# Add these new endpoints to server.py

@app.get("/components/{component_id}/datasheet/status")
async def get_datasheet_status(component_id: str):
    """Get the processing status of a component's datasheet."""
    doc = doc_store.docs.find_one({"component_id": component_id})
    if not doc:
        raise HTTPException(404, "No datasheet found for this component")
        
    doc_id = str(doc["_id"])
    status = processing_status.get(doc_id, {
        "status": doc.get("status", "unknown"),
        "timestamp": doc.get("processed_date")
    })
    
    return {
        "doc_id": doc_id,
        "status": status["status"],
        "timestamp": status["timestamp"]
    }

@app.get("/components/{component_id}/datasheet")
async def get_datasheet_content(component_id: str):
    """Get the processed datasheet content for a component."""
    doc = doc_store.docs.find_one({"component_id": component_id})
    if not doc:
        raise HTTPException(404, "No datasheet found for this component")
        
    doc_id = str(doc["_id"])
    status = processing_status.get(doc_id, {}).get("status")
    
    if status != "completed":
        return {
            "status": status,
            "message": "Datasheet is still being processed"
        }
        
    sections = doc_store.sections.find({"document_id": doc_id})
    if not sections:
        raise HTTPException(404, "No content found for this datasheet")
        
    return {
        "doc_id": doc_id,
        "sections": list(sections)
    }



@app.get("/documents/{doc_id}/processing-status")
async def get_document_processing_status(doc_id: str):
    """Get detailed processing status of a document."""
    try:
        doc = doc_store.docs.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            raise HTTPException(404, "Document not found")
            
        return {
            "doc_id": str(doc["_id"]),
            "status": doc.get("status", "unknown"),
            "section_count": doc.get("section_count", 0),
            "processed_date": doc.get("processed_date"),
            "completed_at": doc.get("completed_at"),
            "error": doc.get("error"),
            "error_at": doc.get("error_at")
        }
    except Exception as e:
        raise HTTPException(500, f"Error fetching status: {str(e)}")
    

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize anything needed on startup."""
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    print("Server started and initialized")

# Shutdown event
# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up resources on shutdown."""
#     vector_store.close()
#     print("Server shutting down, cleaned up resources")

# Update the shutdown event in server.py
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)