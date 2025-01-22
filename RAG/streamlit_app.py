import streamlit as st
import requests
import json
import time
import base64
from typing import Optional, Dict
import os
import fitz  # PyMuPDF
from io import BytesIO

# Constants
API_BASE_URL = "http://localhost:8000"  # Your FastAPI backend URL

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

# Function to upload PDF
def upload_pdf(file) -> Optional[str]:
    """Upload PDF to backend and return document ID"""
    if file is not None:
        files = {"file": (file.name, file, "application/pdf")}
        try:
            response = requests.post(f"{API_BASE_URL}/documents/", files=files)
            response.raise_for_status()
            return response.json()["doc_id"]
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {str(e)}")
            return None
    return None

# Function to check document processing status
def check_processing_status(doc_id: str) -> str:
    """Check document processing status"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{doc_id}/status")
        response.raise_for_status()
        return response.json()["status"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking status: {str(e)}")
        return "error"

# Function to query document
def query_document(doc_id: str, question: str) -> Optional[Dict]:
    """Send query to backend and get response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/documents/{doc_id}/query",
            json={"doc_id": doc_id, "question": question}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying document: {str(e)}")
        return None

def load_pdf_content(uploaded_file):
    """Load PDF content into memory"""
    if uploaded_file is not None:
        try:
            pdf_content = uploaded_file.read()
            st.session_state.pdf_content = pdf_content
            return pdf_content
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
            return None
    return None

# Function to render current PDF page
def render_pdf_page(page_num):
    """Render specific page of PDF"""
    if st.session_state.pdf_content is None:
        return
    
    try:
        # Create PDF document from stored content
        pdf_document = fitz.open(stream=st.session_state.pdf_content, filetype="pdf")
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Scale up for better resolution
        img_bytes = pix.tobytes()
        
        # Display PDF page
        st.image(img_bytes, use_column_width=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀ Previous", key="prev_btn") and page_num > 0:
                st.session_state.current_page = page_num - 1
                st.rerun()
        with col2:
            st.write(f"Page {page_num + 1} of {pdf_document.page_count}")
        with col3:
            if st.button("Next ▶", key="next_btn") and page_num < pdf_document.page_count - 1:
                st.session_state.current_page = page_num + 1
                st.rerun()
                
        pdf_document.close()
    except Exception as e:
        st.error(f"Error rendering PDF page: {str(e)}")

def main():
    st.set_page_config(layout="wide", page_title="DeltaRAG Interface")
    
    # Initialize session state
    init_session_state()
    
    # Title
    st.title("DeltaRAG Document Analysis")
    
    # Create two columns for the layout
    chat_col, pdf_col = st.columns([1, 1])
    
    # Left column - Chat Interface
    with chat_col:
        st.subheader("Chat Interface")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], key="pdf_uploader")
        
        if uploaded_file:
            # Check if this is a new file upload
            if st.session_state.current_doc_id is None:
                # Load PDF content into memory
                pdf_content = load_pdf_content(uploaded_file)
                if pdf_content:
                    # Upload to backend
                    with st.spinner("Uploading document..."):
                        doc_id = upload_pdf(uploaded_file)
                        if doc_id:
                            st.session_state.current_doc_id = doc_id
                            
                            # Wait for processing to complete
                            status = "processing"
                            with st.spinner("Processing document..."):
                                while status == "processing":
                                    status = check_processing_status(doc_id)
                                    if status == "error":
                                        st.error("Error processing document")
                                        break
                                    time.sleep(2)
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if st.session_state.current_doc_id:
            if prompt := st.chat_input("Ask a question about the document"):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get response from backend
                with st.spinner("Thinking..."):
                    response = query_document(st.session_state.current_doc_id, prompt)
                    if response:
                        # Add assistant response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"]
                        })
                st.rerun()
    
    # Right column - PDF Display
    with pdf_col:
        st.subheader("Document Viewer")
        if st.session_state.pdf_content is not None:
            render_pdf_page(st.session_state.current_page)

if __name__ == "__main__":
    main()