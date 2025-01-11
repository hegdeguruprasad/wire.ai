"""
Multimodal RAG System for PDF Processing
--------------------------------------
This module implements a Retrieval Augmented Generation (RAG) system that can process
PDFs containing text, tables, and images. It uses LangChain and various APIs to
extract, process, and generate responses based on document content.

Main components:
1. Document loading and processing
2. Content chunking and structuring
3. Summary generation
4. Vector store integration
5. RAG chain implementation
"""

import os
import uuid
import getpass
import io
import base64
from typing import List, Dict

# Third-party imports
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# Document Processing Functions
def load_document(file_path: str) -> List[Document]:
    """
    Load and process a PDF document using UnstructuredLoader.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of processed pages with headers and footers filtered out
    """
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        partition_via_api=True,
        coordinates=True,
    )

    pages = [page for page in loader.lazy_load()]
    return [page for page in pages if page.metadata.get("category", "").lower() 
            not in ["header", "footer", "pagenumber"]]

def find_title_sections(documents: List[Document]) -> List[Dict]:
    """
    Find and group content by Title sections in the document.
    
    Args:
        documents (List[Document]): List of document objects

    Returns:
        List[Dict]: List of sections containing title and associated content
    """
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

def chunk_content_with_coordinates(section: Dict) -> Dict:
    """
    Chunk text content while maintaining separate coordinates for each page.
    
    Args:
        section (Dict): A section dictionary containing title, content, and metadata

    Returns:
        Dict: Structured dictionary with combined text and page-specific coordinates
    """
    combined_text = []
    points_by_page = {}

    # Process title coordinates
    if 'coordinates' in section['metadata']:
        title_page = section['metadata']['page_number']
        title_points = section['metadata']['coordinates']['points']
        points_by_page[title_page] = title_points.copy()

    # Process content coordinates
    for item in section['content']:
        if item['category'] == 'NarrativeText':
            combined_text.append(item['text'])

        if 'coordinates' in item:
            page_number = item['page_number']
            if page_number not in points_by_page:
                points_by_page[page_number] = []
            points_by_page[page_number].extend(item['coordinates']['points'])

    # Calculate page-specific coordinates
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

def get_image_as_base64(pdf_path: str, page_number: int, coordinates: Dict) -> str:
    """
    Extract an image from a PDF based on coordinates and encode it in base64 format.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (1-based indexing)
        coordinates: Dictionary containing points, layout_width, and layout_height
        
    Returns:
        str: Base64-encoded image string
    """
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_number - 1]
    pix = page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Scale coordinates
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

# Summary Generation Functions
def create_title_text_summary(title: str, text: str, llm) -> str:
    """Generate a summary combining document title and text content."""
    prompt_text = """
    You are an assistant tasked with summarizing document title and text.
    Give a concise summary combining both pieces of information.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Title: {title}
    Text: {text}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": title, "text": text})

def create_table_summary(table_content: str, llm) -> str:
    """Generate a summary of table contents."""
    prompt_text = """
    You are an assistant tasked with summarizing tables.
    Give a concise summary of the table contents.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table content: {content}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": table_content})

def create_image_summary(image_base64: str, llm) -> str:
    """Generate a description of an image."""
    prompt_text = """
    Describe the image in detail. Be specific about any visible elements. 
    The description should be as detailed as possible. 
    The images will be usually of electronic components or circuits.
    Respond only with the description, no additional comment.
    Do not start your message by saying "Here is a description" or anything like that.
    Just give the description as it is.
    """
    messages = [
        ("user", [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

def process_document(document: Dict) -> Dict:
    """
    Process a document section and generate summaries for all its contents.
    
    Args:
        document (Dict): Document section containing title, text, and other content
        
    Returns:
        Dict: Processed document with summaries
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    summary = {
        "title_text_summary": create_title_text_summary(
            document["title"],
            document["combined_text"],
            llm
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
            content_summary["summary"] = create_table_summary(item["text"], llm)
        elif item["category"] == "Image" and item.get("base64"):
            content_summary["summary"] = create_image_summary(item["base64"], llm)
        else:
            continue

        summary["contents"].append(content_summary)

    return summary

# RAG Implementation Functions
def format_docs(documents: List[Dict]) -> Dict:
    """
    Organize document content by type (text, tables, images).
    
    Args:
        documents (List[Dict]): List of document sections
        
    Returns:
        Dict: Organized content by type
    """
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

def build_prompt(kwargs: Dict) -> ChatPromptTemplate:
    """
    Build a prompt combining all content types for the RAG chain.
    
    Args:
        kwargs (Dict): Dictionary containing context and question
        
    Returns:
        ChatPromptTemplate: Formatted prompt template
    """
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

def create_rag_chain(retriever: MultiVectorRetriever):
    """
    Create the RAG chain for processing queries.
    
    Args:
        retriever (MultiVectorRetriever): Configured retriever instance
        
    Returns:
        Chain: Configured RAG chain
    """
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
        | StrOutputParser()
    )
    return chain



def setup_api_keys() -> None:
    """Set up required API keys if not present in environment variables."""
    if "UNSTRUCTURED_API_KEY" not in os.environ:
        os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass("Unstructured API Key:")
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI API Key:")

def initialize_retriever() -> MultiVectorRetriever:
    """
    Initialize the vector store and retriever.
    
    Returns:
        MultiVectorRetriever: Configured retriever instance
    """
    vectorstore = Chroma(
        collection_name="multi_modal_rag", 
        embedding_function=OpenAIEmbeddings()
    )
    store = InMemoryStore()
    id_key = "element_id"

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

def process_pdf_document(file_path: str) -> tuple:
    """
    Process the PDF document and extract all content.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        tuple: (processed_sections, chunked_sections)
    """
    print("Loading and processing document...")
    documents = load_document(file_path)
    sections = find_title_sections(documents)
    chunked_sections = [chunk_content_with_coordinates(section) for section in sections]
    
    # Add unique IDs to content
    print("Processing document sections...")
    for section in chunked_sections:
        for item in section['content']:
            item['unique_id'] = str(uuid.uuid4())
    
    # Process images
    print("Processing images...")
    for section in chunked_sections:
        for content in section.get("content", []):
            if content.get("category") == "Image":
                content["base64"] = get_image_as_base64(
                    file_path,
                    content["page_number"],
                    content["coordinates"]
                )

    # Generate summaries
    print("Generating summaries...")
    processed_sections = [process_document(doc) for doc in chunked_sections]
    
    return processed_sections, chunked_sections

def index_content(retriever: MultiVectorRetriever, 
                 processed_sections: List[Dict], 
                 chunked_sections: List[Dict]) -> None:
    """
    Index all content in the vector store and document store.
    
    Args:
        retriever: Configured MultiVectorRetriever instance
        processed_sections: List of processed document sections
        chunked_sections: List of chunked document sections
    """
    print("Indexing content...")
    id_key = "element_id"
    
    for result in processed_sections:
        # Add section summary to vectorstore
        section_doc = Document(
            page_content=result["title_text_summary"],
            metadata={
                "category": "Section",
                id_key: result["element_id"]
            }
        )
        retriever.vectorstore.add_documents([section_doc])

        # Find corresponding original content
        original_section = next(
            section for section in chunked_sections
            if section['metadata']['element_id'] == result['element_id']
        )

        # Store complete original section in docstore
        retriever.docstore.mset([(result["element_id"], original_section)])

        # Handle individual content items
        for summary_content in result["contents"]:
            content_id = summary_content["unique_id"]

            content_doc = Document(
                page_content=summary_content["summary"],
                metadata={
                    "category": summary_content["category"],
                    id_key: content_id,
                    "parent_section": result["element_id"]
                }
            )
            retriever.vectorstore.add_documents([content_doc])

            original_content = next(
                content for content in original_section['content']
                if content['unique_id'] == content_id
            )
            retriever.docstore.mset([(content_id, original_content)])

def run_interactive_loop(chain: Runnable) -> None:
    """
    Run the interactive question-answering loop.
    
    Args:
        chain: Configured RAG chain
    """
    print("\nDocument processing complete. You can now ask questions about the document.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
            
        try:
            response = chain.invoke(question)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error processing question: {e}")
            print("Please try another question or type 'exit' to quit.")

def main():
    """
    Main function to run the multimodal RAG system.
    
    This function coordinates the document processing pipeline:
    1. Sets up API keys
    2. Initializes the retriever
    3. Processes the input PDF
    4. Indexes content
    5. Handles user queries
    """
    # Setup and initialization
    setup_api_keys()
    retriever = initialize_retriever()
    
    # Get input file
    file_path = input("Enter the path to your PDF file: ")
    
    # Process document
    processed_sections, chunked_sections = process_pdf_document(file_path)
    
    # Index content
    index_content(retriever, processed_sections, chunked_sections)
    
    # Create and run RAG chain
    print("Setting up RAG chain...")
    chain = create_rag_chain(retriever)
    run_interactive_loop(chain)

if __name__ == "__main__":
    main()