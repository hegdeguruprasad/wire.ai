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

# Import your existing RAG system
from deltaRAG import (
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

# Initialize storage systems

doc_store = DocumentStore(
    connection_string="mongodb://localhost:27017",
    database="VectorDatabase",
    docs= "processed_documents",
    sections= "document_sections",
    embeddings= "embeddings_tracking"

)

vector_store = VectorStore(
    dbname="VectorDB",
    user="postgres",
    password="12345",
    host="localhost",
    port="5432"
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


# Global dictionary to store processing status
processing_status = {}
# Add this right after your processing_status dictionary declaration
import logging

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





@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file)
        
        # Check if document already exists
        existing_doc = doc_store.docs.find_one({"file_name": file.filename})
        if existing_doc:
            doc_id = str(existing_doc["_id"])
            # Important: Initialize status for existing documents too
            if doc_id not in processing_status:
                processing_status[doc_id] = {
                    "status": "completed",  # Assume completed for existing docs
                    "timestamp": existing_doc["processed_date"]
                }
            #logger.debug(f"Existing document found. ID: {doc_id}, Status: {processing_status[doc_id]}")
            return DocumentResponse(
                doc_id=doc_id,
                file_name=existing_doc["file_name"],
                processed_date=existing_doc["processed_date"],
                section_count=existing_doc.get("section_count", 0)
            )
        
        # For new documents...
        doc_metadata = {
            "file_name": file.filename,
            "processed_date": datetime.now()
        }
        result = doc_store.docs.insert_one(doc_metadata)
        doc_id = str(result.inserted_id)
        
        # Initialize processing status
        processing_status[doc_id] = {
            "status": "processing",
            "timestamp": datetime.now()
        }
        #logger.debug(f"New document created. ID: {doc_id}, Status: {processing_status[doc_id]}")
        
        # Start background processing
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
        #logger.error(f"Error in upload_document: {str(e)}")
        raise HTTPException(500, f"Error processing document: {str(e)}")





@app.get("/documents/{doc_id}/status", response_model=ProcessingStatus)
async def get_processing_status(doc_id: str):
    """Get the processing status of a document."""
    #logger.debug(f"Checking status for doc_id: {doc_id}")
    #logger.debug(f"Current processing_status: {processing_status}")
    
    # If status not in memory but document exists in MongoDB, initialize it
    if doc_id not in processing_status:
        doc = doc_store.docs.find_one({"_id": ObjectId(doc_id)})
        if doc:
            processing_status[doc_id] = {
                "status": "completed",  # Assume completed for existing docs
                "timestamp": doc["processed_date"]
            }
            #logger.debug(f"Initialized status for existing doc: {processing_status[doc_id]}")
    
    status = processing_status.get(doc_id)
    if not status:
        #logger.debug(f"Document not found for ID: {doc_id}")
        raise HTTPException(404, "Document not found")
    
    return ProcessingStatus(
        doc_id=doc_id,
        status=status["status"],
        timestamp=status["timestamp"]
    )




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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize anything needed on startup."""
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    print("Server started and initialized")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    vector_store.close()
    print("Server shutting down, cleaned up resources")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)