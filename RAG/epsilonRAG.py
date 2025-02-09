# document_processor.py
"""
Multimodal RAG System for PDF Processing
This module implements a Retrieval Augmented Generation (RAG) system that can process
PDFs containing text, tables, and images.
"""

import os
import uuid
import getpass
import io
import base64
from typing import Dict, List, Optional, Tuple, Any, Iterator
from datetime import datetime
import json
import logging
import time

# Third-party imports
import fitz
from PIL import Image
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.schema import Document, BaseStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.messages import HumanMessage
from pymongo import MongoClient
from langchain_community.vectorstores.pgvector import PGVector
import psycopg2
from psycopg2.extras import Json
from langchain_postgres import PGVector  # New import
from psycopg import connect, OperationalError  # Updated import
from psycopg.rows import dict_row
from langchain_postgres import PGVector
from bson import ObjectId  # Add this
import json  # Add this
from tiktoken import get_encoding  # OpenAI token counting

load_dotenv()

# Document Processing
class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def load_document(self) -> List[Document]:
        """Load and process a PDF document."""
        loader = UnstructuredLoader(
            file_path=self.file_path,
            strategy="hi_res",
            partition_via_api=True,
            coordinates=True,
        )
        pages = [page for page in loader.lazy_load()]
        return [page for page in pages if page.metadata.get("category", "").lower() 
                not in ["header", "footer", "pagenumber"]]

    def get_image_as_base64(self, page_number: int, coordinates: Dict) -> str:
        """Extract an image from PDF and encode in base64."""
        pdf_document = fitz.open(self.file_path)
        page = pdf_document[page_number - 1]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        points = coordinates["points"]
        layout_width = coordinates["layout_width"]
        layout_height = coordinates["layout_height"]
        x_scale = pix.width / layout_width
        y_scale = pix.height / layout_height
        
        x1, y1 = points[0][0] * x_scale, points[0][1] * y_scale
        x2, y2 = points[2][0] * x_scale, points[2][1] * y_scale
        
        cropped_image = image.crop((x1, y1, x2, y2))
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        pdf_document.close()
        return base64_image

    def process_document(self) -> tuple:
        """Process PDF document and extract all content."""
        documents = self.load_document()
        sections = self._find_title_sections(documents)
        chunked_sections = [self._chunk_content_with_coordinates(section) 
                          for section in sections]
        
        # Add unique IDs and process images
        for section in chunked_sections:
            for item in section['content']:
                item['unique_id'] = str(uuid.uuid4())
                if item.get("category") == "Image":
                    item["base64"] = self.get_image_as_base64(
                        item["page_number"],
                        item["coordinates"]
                    )

        # Generate summaries
        processed_sections = [self._process_section(doc) for doc in chunked_sections]
        return processed_sections, chunked_sections

    def _find_title_sections(self, documents: List[Document]) -> List[Dict]:
        """Find and group content by Title sections."""
        sections = []
        current_section = None

        for doc in documents:
            metadata = doc.metadata
            category = metadata.get("category", "")

            if category == "Title":
                if current_section:
                    sections.append(current_section)

                current_section = {
                    "title": doc.page_content.strip(),
                    "content": [],
                    "metadata": {
                        "page_number": metadata.get("page_number"),
                        "coordinates": metadata.get("coordinates"),
                        "source": metadata.get("source"),
                        "element_id": metadata.get("element_id")
                    }
                }
            elif current_section is not None:
                content_item = {
                    "text": doc.page_content.strip(),
                    "category": category,
                    "coordinates": metadata.get("coordinates"),
                    "page_number": metadata.get("page_number")
                }
                current_section["content"].append(content_item)

        if current_section:
            sections.append(current_section)

        return sections

    def _chunk_content_with_coordinates(self, section: Dict) -> Dict:
        """Chunk text content while maintaining coordinates."""
        combined_text = []
        points_by_page = {}

        if 'coordinates' in section['metadata']:
            title_page = section['metadata']['page_number']
            title_points = section['metadata']['coordinates']['points']
            points_by_page[title_page] = title_points.copy()

        for item in section['content']:
            if item['category'] == 'NarrativeText':
                combined_text.append(item['text'])

            if 'coordinates' in item:
                page_number = item['page_number']
                if page_number not in points_by_page:
                    points_by_page[page_number] = []
                points_by_page[page_number].extend(item['coordinates']['points'])

        coordinates_by_page = self._calculate_coordinates(points_by_page)

        return {
            'title': section['title'],
            'combined_text': ' '.join(combined_text),
            'coordinates_by_page': coordinates_by_page,
            'content': [
                {**item, 'text': item['text']}
                for item in section['content']
                if item['category'] != 'NarrativeText'
            ],
            'metadata': {
                'page_number': section['metadata']['page_number'],
                'coordinates': coordinates_by_page,
                'source': section['metadata']['source'],
                'element_id': section['metadata']['element_id']
            }
        }

    def _calculate_coordinates(self, points_by_page: Dict) -> Dict:
        """Calculate page-specific coordinates."""
        coordinates_by_page = {}
        for page_number, points in points_by_page.items():
            if points:
                left = min(p[0] for p in points)
                right = max(p[0] for p in points)
                top = min(p[1] for p in points)
                bottom = max(p[1] for p in points)

                coordinates_by_page[page_number] = {
                    'points': [[left, top], [left, bottom], 
                              [right, bottom], [right, top]],
                    'system': 'PixelSpace',
                    'layout_width': 1700,
                    'layout_height': 2200
                }
        return coordinates_by_page

    def _process_section(self, document: Dict) -> Dict:
        """Process a document section and generate summaries."""
        summary = {
            "title_text_summary": self._create_title_text_summary(
                document["title"],
                document["combined_text"]
            ),
            "element_id": document["metadata"]["element_id"],
            "contents": []
        }

        for item in document["content"]:
            content_summary = {
                "category": item["category"],
                "unique_id": item["unique_id"]
            }

            if item["category"] == "Table":
                content_summary["summary"] = self._create_table_summary(item["text"])
            elif item["category"] == "Image" and item.get("base64"):
                content_summary["summary"] = self._create_image_summary(item["base64"])
            else:
                continue

            summary["contents"].append(content_summary)

        return summary

    def _create_title_text_summary(self, title: str, text: str) -> str:
        """Generate summary combining document title and text."""
        prompt = ChatPromptTemplate.from_template("""
            Summarize this document section combining title and text.
            Title: {title}
            Text: {text}
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"title": title, "text": text})

    def _create_table_summary(self, table_content: str) -> str:
        """Generate summary of table contents."""
        prompt = ChatPromptTemplate.from_template("""
            Give a concise summary of the table contents:
            {content}
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"content": table_content})

    def _create_image_summary(self, image_base64: str) -> str:
        """Generate description of an image."""
        messages = [
            ("user", [
                {"type": "text", "text": "Describe this image in detail:"},
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

# Storage Classes
class MongoDocStore(BaseStore[str, Dict]):
    """MongoDB-based document store."""
    
    def __init__(self, collection):
        self.collection = collection

    def mget(self, keys: List[str]) -> List[Optional[Dict]]:
        results = []
        for key in keys:
            doc = self.collection.find_one({"_id": key})
            if doc:
                doc_dict = {k: v for k, v in doc.items() if k != '_id'}
                results.append(doc_dict)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: List[Tuple[str, Dict]]) -> None:
        for key, value in key_value_pairs:
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Value must be JSON serializable: {e}")
            self.collection.update_one(
                {"_id": key},
                {"$set": value},
                upsert=True
            )

    def mdelete(self, keys: List[str]) -> None:
        self.collection.delete_many({"_id": {"$in": keys}})
    
    def yield_keys(self) -> Iterator[str]:
        cursor = self.collection.find({}, {"_id": 1})
        for doc in cursor:
            yield str(doc["_id"])

class DocumentStore:
    """Handle storage of processed PDF documents."""
    
    def __init__(self, connection_string: str, database: str, docs:str,  sections: str, embeddings: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.docs = self.db[docs]
        self.sections = self.db[sections]
        # Add embeddings collection to track which elements have embeddings
        self.embeddings = self.db[embeddings]
        print("Connection to MongoDB successful")

    def _sanitize_for_mongo(self, data: Any) -> Any:
        """Convert data to MongoDB-safe format."""
        if isinstance(data, dict):
            return {
                str(k): self._sanitize_for_mongo(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_for_mongo(item) for item in data]
        elif isinstance(data, (ObjectId, datetime)):
            return str(data)
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)

    def store_document(self, file_path: str, processed_sections: List[Dict], 
                      chunked_sections: List[Dict], doc_id: str = None) -> str:
        """Store document with improved error handling."""
        try:
            if doc_id:
                # Update existing document
                self.docs.update_one(
                    {"_id": ObjectId(doc_id)},
                    {
                        "$set": {
                            "file_name": os.path.basename(file_path),
                            "processed_date": datetime.now(),
                            "section_count": len(processed_sections),
                            "status": "processing"
                        }
                    }
                )
            else:
                # Create new document if no doc_id provided
                doc_metadata = {
                    "file_name": os.path.basename(file_path),
                    "processed_date": datetime.now(),
                    "section_count": len(processed_sections),
                    "status": "processing"
                }
                result = self.docs.insert_one(doc_metadata)
                doc_id = str(result.inserted_id)

            self._store_sections(doc_id, processed_sections, chunked_sections)
            return doc_id
            
        except Exception as e:
            logging.error(f"Error storing document: {str(e)}")
            raise

    def _store_metadata(self, file_path: str, processed_sections: List[Dict]) -> str:
        doc_metadata = {
            "file_name": os.path.basename(file_path),
            "processed_date": datetime.now(),
            "section_count": len(processed_sections)
        }
        result = self.docs.insert_one(doc_metadata)
        return str(result.inserted_id)

    def _store_sections(self, doc_id: str, processed_sections: List[Dict],
                       chunked_sections: List[Dict]) -> None:
        """Store document sections with improved type handling."""
        try:
            # First, remove any existing sections for this document
            self.sections.delete_many({"document_id": doc_id})
            
            for proc_section, chunk_section in zip(processed_sections, chunked_sections):
                sanitized_data = {
                    "document_id": doc_id,
                    "processed_data": self._sanitize_for_mongo(proc_section),
                    "original_chunk": self._sanitize_for_mongo(chunk_section),
                    "stored_date": datetime.now().isoformat()
                }

                if not self._validate_structure(sanitized_data):
                    logging.error("Invalid document structure:")
                    logging.error(f"Document: {sanitized_data}")
                    raise ValueError("Invalid document structure")

                self.sections.insert_one(sanitized_data)
                
        except Exception as e:
            logging.error(f"Error storing section: {str(e)}")
            raise ValueError(f"Failed to store section: {str(e)}")

    def _validate_structure(self, document: Dict) -> bool:
        """More permissive validation for server context."""
        def check_keys(obj):
            if isinstance(obj, dict):
                # Allow ObjectId and other BSON types
                return all(isinstance(k, (str, ObjectId)) and check_keys(v) 
                          for k, v in obj.items())
            elif isinstance(obj, list):
                return all(check_keys(item) for item in obj)
            elif isinstance(obj, (str, int, float, bool, datetime, ObjectId, type(None))):
                return True
            return False  # Block other types
        
        try:
            return check_keys(document)
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return False

    def get_document_sections(self, doc_id: str) -> tuple:
        """Retrieve processed sections for a document."""
        sections = list(self.sections.find({"document_id": doc_id}))
        if not sections:
            return None, None
        return ([s["processed_data"] for s in sections], 
                [s["original_chunk"] for s in sections])
    
    def has_embeddings(self, doc_id: str, element_id: str) -> bool:
        """Check if element already has embeddings stored."""
        return bool(self.embeddings.find_one({
            "document_id": doc_id,
            "element_id": element_id
        }))

    def track_embedding(self, doc_id: str, element_id: str, category: str):
        """Track that an element has embeddings stored."""
        self.embeddings.update_one(
            {
                "document_id": doc_id,
                "element_id": element_id
            },
            {
                "$set": {
                    "category": category,
                    "created_at": datetime.now()
                }
            },
            upsert=True
        )

class VectorStore:
    """Handle storage of vector embeddings using Supabase PostgreSQL."""

    def __init__(self, db_url: str, collection_name: str = "pdf_vectors"):
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                self.store = PGVector(
                    collection_name=collection_name,
                    connection=db_url,
                    embeddings=OpenAIEmbeddings(),
                    use_jsonb=True
                )
                print("Connection to Supabase PostgreSQL successful")
                break
            except OperationalError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed to connect after {max_retries} attempts: {e}")
                    raise
                print(f"Connection attempt {retry_count} failed, retrying in 5 seconds...")
                time.sleep(5)

    def close(self):
        """Close database connection."""
        if hasattr(self, 'store'):
            self.store.connection.close()

def convert_keys_to_strings(d):
    """Recursively convert dictionary keys to strings."""
    if isinstance(d, dict):
        return {str(k): convert_keys_to_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_strings(i) for i in d]
    else:
        return d

# RAG Implementation
class RAGSystem:
    def __init__(self, doc_store: DocumentStore, vector_store: VectorStore):
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)

    def setup_retriever(self, processed_sections: List[Dict],
                   chunked_sections: List[Dict],
                   doc_id: str) -> MultiVectorRetriever:
        """Set up retriever with MongoDB and PostgreSQL integration."""
        mongo_store = MongoDocStore(self.doc_store.sections)
        retriever = MultiVectorRetriever(
            vectorstore=self.vector_store.store,
            docstore=mongo_store,
            id_key="element_id",
            search_kwargs={"k": 5}
        )
        
        self._store_vectors(retriever, processed_sections, chunked_sections, doc_id)
        return retriever

    def _store_vectors(self, retriever: MultiVectorRetriever,
                      processed_sections: List[Dict],
                      chunked_sections: List[Dict],
                      doc_id: str) -> None:
        """Store vectors in PostgreSQL and documents in MongoDB."""

        id_key = "element_id"
        
        for result in processed_sections:
            element_id = result["element_id"]
            
            # Check if section vectors already exist
            if not self.doc_store.has_embeddings(doc_id, element_id):
                # Store section vectors
                section_doc = Document(
                    page_content=result["title_text_summary"],
                    metadata={
                        "category": "Section",
                        id_key: element_id
                    }
                )
                self.vector_store.store.add_documents([section_doc])
                self.doc_store.track_embedding(doc_id, element_id, "Section")
            
            # Find and store original section
            original_section = next(
                section for section in chunked_sections
                if section['metadata']['element_id'] == element_id
            )
            cleaned_section = convert_keys_to_strings(original_section)
            retriever.docstore.mset([(element_id, cleaned_section)])
            
            # Store content items
            for summary_content in result["contents"]:
                content_id = summary_content["unique_id"]
                
                # Check if content vectors already exist
                if not self.doc_store.has_embeddings(doc_id, content_id):
                    content_doc = Document(
                        page_content=summary_content["summary"],
                        metadata={
                            "category": summary_content["category"],
                            id_key: content_id,
                            "parent_section": element_id
                        }
                    )
                    self.vector_store.store.add_documents([content_doc])
                    self.doc_store.track_embedding(
                        doc_id, content_id, summary_content["category"]
                    )
                
                original_content = next(
                    content for content in original_section['content']
                    if content['unique_id'] == content_id
                )
                retriever.docstore.mset([(content_id, original_content)])

    def process_question(self, retriever: MultiVectorRetriever, question: str) -> str:
        """Retrieves top 5 relevant sections and ensures token limit compliance."""

        retrieved_results = retriever.invoke(question)

        # Extract text from valid documents
        relevant_docs = []
        for doc in retrieved_results:
            if isinstance(doc, dict):
                if "content" in doc:  # If "content" key exists, use it
                    text = doc["content"]
                elif "text" in doc:  # If "text" key exists (tables, figures, etc.), use it
                    text = doc["text"]
                else:
                    print("⚠️ Warning: Retrieved document is missing text content:", doc)
                    continue  # Skip invalid documents

                # Ensure extracted text is a string
                if isinstance(text, list):  # Convert list to string if needed
                    text = " ".join(map(str, text))

                relevant_docs.append(text)
            elif isinstance(doc, str):  # If it's already a string, keep it
                relevant_docs.append(doc)
            else:
                print("⚠️ Warning: Retrieved document is an unknown type:", type(doc))

        # Combine retrieved chunks into a single formatted context string
        full_context = "\n\n".join(relevant_docs)

        # Limit context size to prevent exceeding token limit
        trimmed_context = self._trim_context(full_context, max_tokens=8000)

        # Generate response using GPT-4o-mini
        response = self.llm.invoke(f"Context:\n{trimmed_context}\n\nQuestion: {question}")

        return response

    def _trim_context(self, text: str, max_tokens: int = 8000) -> str:
        """Trims the context to fit within a safe token limit."""
        enc = get_encoding("cl100k_base")  # Use OpenAI tokenizer
        tokens = enc.encode(text)

        # Keep only the first `max_tokens`
        trimmed_tokens = tokens[:max_tokens]
        return enc.decode(trimmed_tokens)

    def _format_docs(self, documents: List[Dict]) -> Dict:
        """Organize document content by type."""
        texts = []
        tables = []
        images = []

        for doc in documents:
            if doc.get('combined_text'):
                texts.append({
                    'title': doc.get('title', ''),
                    'content': doc['combined_text'],
                })

            for item in doc.get('content', []):
                if item['category'] == 'Table':
                    tables.append({
                        'content': item['text'],
                        'page': item['page_number']
                    })
                elif item['category'] == 'Image':
                    images.append({
                        'base64': item['base64']
                    })

        return {
            "texts": texts,
            "tables": tables,
            "images": images
        }

    def _build_prompt(self, kwargs: Dict) -> ChatPromptTemplate:
        """Build prompt combining all content types."""
        docs = kwargs["context"]
        question = kwargs["question"]
        
        context_parts = []
        
        for text in docs["texts"]:
            if text['title']:
                context_parts.append(f"Title: {text['title']}")
            context_parts.append(f"Content: {text['content']}")

        for table in docs["tables"]:
            context_parts.append(f"\nTable (Page {table['page']}):\n{table['content']}")

        for content in docs["images"]:
            context_parts.append(f"\nImage URL: data:image/jpeg;base64,{content['base64']}")

        context_text = "\n".join(context_parts)
        
        prompt_content = [{
            "type": "text",
            "text": f"""Please analyze the following technical document content and answer the question.

Context:
{context_text}

Question: {question}

Please provide a detailed answer based on the information given above."""
        }]

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

# def setup_api_keys() -> None:
#     """Set up required API keys."""
#     if "UNSTRUCTURED_API_KEY" not in os.environ:
#         os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass("Unstructured API Key:")
#     if "OPENAI_API_KEY" not in os.environ:
#         os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI API Key:")

def setup_api_keys() -> None:
    """Check for required API keys."""
    required_keys = ["UNSTRUCTURED_API_KEY", "OPENAI_API_KEY"]
    for key in required_keys:
        if key not in os.environ:
            raise EnvironmentError(f"Missing required environment variable: {key}")

def get_valid_file_path() -> str:
    """Get and validate PDF file path."""
    while True:
        file_path = input("Enter the path to your PDF file: ")
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            return file_path
        print("Invalid path. Please enter a valid PDF file path.")

def run_interactive_loop(rag_system: RAGSystem, retriever: MultiVectorRetriever) -> None:
    """Run interactive question-answering loop."""
    print("\nDocument processing complete. You can now ask questions.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
            
        try:
            response = rag_system.process_question(retriever, question)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error processing question: {e}")
            print("Please try another question or type 'exit' to quit.")

def main():
    """Main function to run the RAG system."""
    setup_api_keys()

    # Initialize storage
    doc_store = DocumentStore(
        connection_string="mongodb://localhost:27017",
        database="VectorDatabase",
        docs= "processed_documents",
        sections= "document_sections",
        embeddings= "embeddings_tracking"

    )

    # vector_store = VectorStore(
    #     dbname="VectorDB",
    #     user="postgres",
    #     password="12345",
    #     host="localhost",
    #     port="5432"
    # )
    
    vector_store = VectorStore(
        # db_url="postgresql://postgres.qcqeeulzbgezcrkvrhoh:WireAi#12345$@aws-0-us-west-1.pooler.supabase.com:6543/postgres",
        db_url= "postgresql://postgres.qcqeeulzbgezcrkvrhoh:9CkzjF0rXdgmF3bV@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
        collection_name="pdf_vectors"
    )

    # Get and process document
    file_path = get_valid_file_path()
    file_name = os.path.basename(file_path)
    
    # Check if document exists
    doc = doc_store.docs.find_one({"file_name": file_name})
    if doc:
        print("Loading previously processed document...")
        doc_id = str(doc["_id"])
        processed_sections, chunked_sections = doc_store.get_document_sections(doc_id)
        
        if not processed_sections or not chunked_sections:
            print("Processing document...")
            processor = DocumentProcessor(file_path)
            processed_sections, chunked_sections = processor.process_document()
            doc_id = doc_store.store_document(file_path, processed_sections, chunked_sections)
    else:
        print("Processing new document...")
        processor = DocumentProcessor(file_path)
        processed_sections, chunked_sections = processor.process_document()
        doc_id = doc_store.store_document(file_path, processed_sections, chunked_sections)
    
    if not processed_sections or not chunked_sections:
        raise ValueError("Failed to get processed document sections")
    
    # Set up RAG system
    rag_system = RAGSystem(doc_store, vector_store)
    retriever = rag_system.setup_retriever(processed_sections, chunked_sections,doc_id)
    
    # Run interactive loop
    run_interactive_loop(rag_system, retriever)
    
    # Cleanup
    vector_store.close()

if __name__ == "__main__":
    main()