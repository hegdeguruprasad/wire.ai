# RAG System API Server

A FastAPI-based server that provides API endpoints for a RAG (Retrieval-Augmented Generation) system. This server allows users to upload PDF documents, process them, and query them using natural language questions.

## Features

- PDF document upload and processing
- Asynchronous document processing
- Document status tracking
- Natural language querying of processed documents
- Vector storage for efficient retrieval
- Document management (list, delete)

## Prerequisites

- Python 3.8+
- MongoDB
- PostgreSQL
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install fastapi uvicorn python-multipart pymongo psycopg2-binary
```

3. Configure your database connections in `server.py`:
```python
doc_store = DocumentStore(
    connection_string="mongodb://localhost:27017",
    database="VectorDatabase",
    docs="processed_documents",
    sections="document_sections",
    embeddings="embeddings_tracking"
)

vector_store = VectorStore(
    dbname="VectorDB",
    user="postgres",
    password="12345",
    host="localhost",
    port="5432"
)
```

## Running the Server

Start the server using uvicorn:
```bash
uvicorn server:app --reload
```

The server will start at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

## API Endpoints

### 1. Upload Document
Upload a PDF document for processing.

```bash
curl --location 'http://localhost:8000/documents/' \
--header 'accept: application/json' \
--form 'file=@"/path/to/your/document.pdf"'
```

Response:
```json
{
    "doc_id": "67839e56930625ace3f8a2db",
    "file_name": "document.pdf",
    "processed_date": "2025-01-12T16:19:58.456000",
    "section_count": 0
}
```

### 2. Check Processing Status
Check the processing status of an uploaded document.

```bash
curl --location 'http://localhost:8000/documents/67839e56930625ace3f8a2db/status' \
--header 'accept: application/json'
```

Response:
```json
{
    "doc_id": "67839e56930625ace3f8a2db",
    "status": "completed",
    "timestamp": "2025-01-12T16:19:58.456000"
}
```

### 3. Query Document
Ask questions about a processed document.

```bash
curl --location 'http://localhost:8000/documents/67839e56930625ace3f8a2db/query' \
--header 'Content-Type: application/json' \
--header 'accept: application/json' \
--data '{
    "doc_id": "67839e56930625ace3f8a2db",
    "question": "What is the main topic of this document?"
}'
```

Response:
```json
{
    "doc_id": "67839e56930625ace3f8a2db",
    "question": "What is the main topic of this document?",
    "answer": "The response from the RAG system..."
}
```

### 4. List Documents
Get a list of all processed documents.

```bash
curl --location 'http://localhost:8000/documents/' \
--header 'accept: application/json'
```

Response:
```json
[
    {
        "doc_id": "67839e56930625ace3f8a2db",
        "file_name": "document.pdf",
        "processed_date": "2025-01-12T16:19:58.456000",
        "section_count": 134
    }
]
```

### 5. Delete Document
Delete a document and its associated data.

```bash
curl --location --request DELETE 'http://localhost:8000/documents/67839e56930625ace3f8a2db' \
--header 'accept: application/json'
```

Response:
```json
{
    "message": "Document deleted successfully"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (e.g., invalid file type)
- 404: Document Not Found
- 500: Internal Server Error

Example error response:
```json
{
    "detail": "Error message here"
}
```

## Project Structure

```
.
├── server.py           # Main FastAPI server file
├── deltaRAG/          # RAG system implementation
├── uploads/           # Directory for uploaded files
└── README.md          # This file
```

## Development

For development, the server includes:
- CORS middleware enabled for all origins (customize for production)
- Automatic API documentation
- Debug logging
- Background task processing

## Production Considerations

Before deploying to production:
1. Configure proper CORS settings
2. Set up proper database security
3. Implement authentication
4. Configure proper logging
5. Set up error monitoring
6. Configure proper file storage
7. Set up proper API rate limiting