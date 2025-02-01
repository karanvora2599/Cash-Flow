# main.py

import os
import uuid
import logging
import time
from typing import Dict, Any

import openai
from openai import OpenAIError  # Correctly import OpenAIError
from pinecone import Pinecone, ServerlessSpec  # Updated import
from pinecone.exceptions import PineconeException  # Import PineconeException
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Load environment variables from a .env file if present
from dotenv import load_dotenv
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named app.log
        logging.StreamHandler()          # Also log to console
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Text to Pinecone Vector Store API")

# Initialize OpenAI and Pinecone API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # e.g., 'us-west1-gcp'

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.critical("API keys and Pinecone environment must be set as environment variables.")
    raise ValueError("API keys and Pinecone environment must be set as environment variables.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

try:
    # Initialize Pinecone using the new Pinecone class
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    logger.info("Pinecone initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Pinecone: {e}", exc_info=True)
    raise

# Pydantic Models
class TextInput(BaseModel):
    text: str
    chunk_size: int = 500
    overlap: int = 50

class VectorStoreResponse(BaseModel):
    index_name: str
    dimension: int
    metadata: Dict[str, Any]

class DeleteVectorStoreRequest(BaseModel):
    index_name: str

class DeleteVectorStoreResponse(BaseModel):
    message: str
    index_name: str

# Custom Exception Classes
class PineconeIndexNotFoundError(HTTPException):
    def __init__(self, index_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pinecone index '{index_name}' not found."
        )

class PineconeServiceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail
        )

class OpenAIServiceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail
        )

# Helper Functions
def retry_operation(operation, retries=3, delay=2):
    """
    Retry an operation multiple times with delay.
    """
    for attempt in range(retries):
        try:
            return operation()
        except (PineconeException, openai.APIError) as e:  # Updated exception
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"All {retries} attempts failed.")
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="External service is unavailable. Please try again later."
    )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into chunks with specified size and overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logger.debug(f"Text split into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(text_chunks):
    """
    Generate embeddings for each text chunk using OpenAI.
    """
    embeddings = []
    try:
        # Batch process all chunks at once
        response = openai_client.embeddings.create(
            input=text_chunks,
            model="text-embedding-ada-002"
        )
        embeddings = [embedding.embedding for embedding in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings successfully.")
        return embeddings
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        raise OpenAIServiceError("Failed to generate embeddings.")
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during embedding generation."
        )

# Exception Handler for Validation Errors
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# API Endpoints
@app.post("/create-vector-store", response_model=VectorStoreResponse)
def create_vector_store(input: TextInput):
    logger.info("Received request to create a new vector store.")
    try:
        # Step 1: Chunk the text
        text_chunks = chunk_text(input.text, input.chunk_size, input.overlap)
        logger.info(f"Text split into {len(text_chunks)} chunks.")

        # Step 2: Generate embeddings with retry
        def generate():
            return generate_embeddings(text_chunks)
        embeddings = retry_operation(generate)

        # Step 3: Create a unique index name
        unique_id = str(uuid.uuid4()).replace("-", "")[:12]  # Take first 12 characters of UUID
        index_name = f"vecstore{unique_id}".lower()  # Ensure all lowercase
        logger.debug(f"Generated valid index name: {index_name}")

        # Step 4: Create Pinecone index with retry
        def create_pinecone_index():
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',  # Recommended for text embeddings
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Created Pinecone index: {index_name}")
            else:
                logger.warning(f"Index {index_name} already exists")
        retry_operation(create_pinecone_index)

        index = pc.Index(index_name)

        # Step 5: Prepare data for Pinecone
        vectors = [
                    (
                        f"chunk_{i}",  # Unique ID
                        embedding,      # The embedding vector
                        {"text": chunk} # Metadata
                    )
                    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))
                ]
        logger.debug(f"Prepared {len(vectors)} vectors for upsert.")

        # Step 6: Upsert vectors into Pinecone with retry
        def upsert_vectors():
            index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} vectors into '{index_name}'.")
        retry_operation(upsert_vectors)

        # Step 7: Prepare metadata to return
        metadata = {
            "index_name": index_name,
            "environment": PINECONE_ENVIRONMENT,
            "dimension": 1536,
            "vector_count": len(vectors)
        }

        return VectorStoreResponse(
            index_name=index_name,
            dimension=1536,
            metadata=metadata
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he

    except Exception as e:
        logger.error(f"Unexpected error in create_vector_store: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while creating the vector store.")

@app.post("/delete-vector-store", response_model=DeleteVectorStoreResponse)
def delete_vector_store(request: DeleteVectorStoreRequest):
    index_name = request.index_name
    logger.info(f"Received request to delete Pinecone index: {index_name}")
    
    try:
        def check_index_exists():
            if index_name not in pc.list_indexes().names():
                logger.warning(f"Pinecone index '{index_name}' not found.")
                raise PineconeIndexNotFoundError(index_name)
        retry_operation(check_index_exists)

        # Step 1: Delete the index with retry
        def delete_pinecone_index():
            pc.delete_index(index_name)
            logger.info(f"Pinecone index '{index_name}' successfully deleted.")
        retry_operation(delete_pinecone_index)

        return DeleteVectorStoreResponse(
            message=f"Pinecone index '{index_name}' has been deleted successfully.",
            index_name=index_name
        )
    
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he
    
    except Exception as e:
        logger.error(f"Unexpected error while deleting index '{index_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while deleting the index.")