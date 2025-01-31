import os
import uuid
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict
import time

import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec  # Updated import
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator

from cerebras.cloud.sdk import Cerebras
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import ConversationalRetrievalChain

import Prompts

from dotenv import load_dotenv
load_dotenv()

# Configure logging
LOG_DIR = os.path.abspath("logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "RDF.log")

# Create handlers
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

# Create formatters with more context
detailed_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
)

file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(detailed_formatter)

# Configure root logger
logger = logging.getLogger("RDF")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(detailed_formatter)
logger.addHandler(console_handler)

app = FastAPI()

# =======================
# Initialize Cerebras Client
# =======================

# Fetch API key from environment variables
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    logger.error("CEREBRAS_API_KEY environment variable not set")
    raise EnvironmentError("CEREBRAS_API_KEY environment variable not set")

try:
    client = Cerebras(api_key=CEREBRAS_API_KEY)
    logger.info("Cerebras client initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Cerebras client")
    raise e

# =======================
# In-Memory Session Storage
# =======================

# Note: For production, replace with a persistent database
sessions: Dict[str, Dict] = {}

# =======================
# Initialize Pinecone
# =======================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # e.g., 'us-west1-gcp'

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.critical("PINECONE_API_KEY and PINECONE_ENV environment variables must be set.")
    raise EnvironmentError("PINECONE_API_KEY and PINECONE_ENV environment variables must be set.")

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

# =======================
# Pydantic Models for Session
# =======================

class SessionStartRequest(BaseModel):
    # Optional: Add any parameters needed for session initialization
    pass

class SessionStartResponse(BaseModel):
    session_id: str
    
# =======================
# Pydantic Models for Chat
# =======================

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    input: str
    response: str
    
# =======================
# Pydantic Models for Delete Chat
# =======================
class DeleteRequest(BaseModel):
    session_id: str

class DeleteResponse(BaseModel):
    detail: str
    session_id: str
    remaining_sessions: int
    
# =======================
# Pydantic Models for RAG Chat
# =======================
    
class ChatRagRequest(BaseModel):
    session_id: str
    vectorstore_id: str
    message: str

class ChatRagResponse(BaseModel):
    session_id: str
    vectorstore_id: str
    input: str
    response: str

# =======================
# Custom Exception Classes
# =======================

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
        
# =======================
# Helper Functions
# =======================

def retry_operation(operation, retries=3, delay=2):
    """
    Retry an operation multiple times with delay.
    """
    for attempt in range(retries):
        try:
            return operation()
        except (pinecone.PineconeException, openai.error.OpenAIError) as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"All {retries} attempts failed.")
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="External service is unavailable. Please try again later."
    )

# def load_vectorstore(vectorstore_id: str):
#     """
#     Load a Pinecone vector store based on the vectorstore_id.
#     """
#     try:
#         if vectorstore_id not in pinecone.list_indexes():
#             logger.error(f"Pinecone index '{vectorstore_id}' does not exist.")
#             raise PineconeIndexNotFoundError(vectorstore_id)
        
#         embeddings = OpenAIEmbeddings()
#         vector_store = Pinecone.from_existing_index(vectorstore_id, embeddings)
#         logger.info(f"Pinecone vector store '{vectorstore_id}' loaded successfully.")
#         return vector_store
#     except PineconeIndexNotFoundError:
#         raise
#     except pinecone.PineconeException as e:
#         logger.error(f"Pinecone service error while loading index '{vectorstore_id}': {e}", exc_info=True)
#         raise PineconeServiceError("Failed to load Pinecone vector store.")
#     except Exception as e:
#         logger.exception(f"Unexpected error while loading vector store '{vectorstore_id}': {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load vector store.")

def load_vectorstore(vectorstore_id: str):
    """
    Load a Pinecone vector store based on the vectorstore_id.
    """
    try:
        if vectorstore_id not in pc.list_indexes():
            logger.error(f"Pinecone index '{vectorstore_id}' does not exist.")
            raise PineconeIndexNotFoundError(vectorstore_id)
        
        embeddings = OpenAIEmbeddings()
        vector_store = LangchainPinecone.from_existing_index(
            index_name=vectorstore_id,
            embedding=embeddings,
            client=pc  # Pass the Pinecone client instance if required
        )
        logger.info(f"Pinecone vector store '{vectorstore_id}' loaded successfully.")
        return vector_store
    except PineconeIndexNotFoundError:
        raise
    except pinecone.PineconeException as e:
        logger.error(f"Pinecone service error while loading index '{vectorstore_id}': {e}", exc_info=True)
        raise PineconeServiceError("Failed to load Pinecone vector store.")
    except Exception as e:
        logger.exception(f"Unexpected error while loading vector store '{vectorstore_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load vector store.")

def create_qa_chain(vector_store):
    """
    Create a ConversationalRetrievalChain with the provided vector store.
    """
    try:
        # Initialize conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize Cerebras LLM
        cerebras_llm = CerebrasLLM(client=client, model="llama3.1-8b")
        
        # Initialize RAG pipeline
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=cerebras_llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )
        logger.info("ConversationalRetrievalChain initialized successfully.")
        return qa_chain
    except Exception as e:
        logger.exception(f"Failed to create QA chain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initialize QA chain.")
    
# =======================
# Cerebras LLM Wrapper
# =======================

class CerebrasLLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": Prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            ai_response = response.choices[0].message.content
            logger.debug(f"Cerebras AI response: {ai_response}")
            return ai_response
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise OpenAIServiceError("Failed to generate AI response.")
        except Exception as e:
            logger.exception(f"Unexpected error in CerebrasLLM: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate AI response.")


# =======================
# Exception Handling
# =======================

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
    logger.exception(f"Unhandled exception for request {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"},
    )

# =======================
# API Endpoints
# =======================

# =======================
# Start-Session Endpoint
# =======================
@app.post("/start-session", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Initializes a new chat session with a unique ID.
    """
    try:
        session_id = str(uuid.uuid4())
        memory = ConversationBufferMemory()
        memory.chat_memory.add_ai_message(Prompts.SYSTEM_PROMPT)
        sessions[session_id] = {
            "memory": memory
        }
        logger.info(f"Started new session with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        logger.exception("Error starting new session")
        raise HTTPException(status_code=500, detail="Failed to start session")
    
# =======================
# Chat Endpoint
# =======================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Processes a chat message and returns an AI response.
    """
    session_id = request.session_id
    user_message = request.message

    logger.debug(f"Received message for session {session_id}: {user_message}")

    # Validate session ID
    if session_id not in sessions:
        logger.warning(f"Session ID not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    memory = sessions[session_id]["memory"]

    try:
        # Add user message to memory
        memory.chat_memory.add_user_message(user_message)
        logger.debug(f"Added user message to session {session_id} memory")

        # Get conversation history
        history = memory.load_memory_variables({}).get("history", "")
        logger.debug(f"Loaded history for session {session_id}: {history}")

        # Create prompt with history
        prompt = f"{history}\nUser: {user_message}\nAI:"
        logger.debug(f"Generated prompt for Cerebras: {prompt}")

        # Get AI response from Cerebras
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": Prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model="llama3.1-8b",
        )

        ai_response = response.choices[0].message.content
        logger.debug(f"Received AI response for session {session_id}: {ai_response}")

        # Add AI response to memory
        memory.chat_memory.add_ai_message(ai_response)
        logger.debug(f"Added AI message to session {session_id} memory")

        return {
            "session_id": session_id,
            "input": user_message,
            "response": ai_response
        }

    except HTTPException as http_exc:
        # Specific HTTP exceptions are logged at warning level
        logger.warning(f"HTTPException for session {session_id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # General exceptions are logged with stack trace
        logger.exception(f"Error processing chat for session {session_id}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")
    
# =======================
# RAG Chat Endpoint
# =======================
@app.post("/chat-rag", response_model=ChatRagResponse)
async def chat_rag_endpoint(request: ChatRagRequest):
    """
    Processes a chat message with RAG using a specified vector store and returns an AI response.
    """
    session_id = request.session_id
    vectorstore_id = request.vectorstore_id
    user_message = request.message

    logger.debug(f"Received RAG message for session {session_id} with vectorstore_id {vectorstore_id}: {user_message}")

    # Validate session ID
    if session_id not in sessions:
        logger.warning(f"Session ID not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    memory = sessions[session_id]["memory"]

    try:
        # Load the specified vector store with retry
        vector_store = retry_operation(lambda: load_vectorstore(vectorstore_id))

        # Create QA chain with the loaded vector store
        qa_chain = create_qa_chain(vector_store)

        # Update the session's memory with the QA chain's memory
        # This ensures that the conversation history is maintained
        # You may need to adjust based on how your ConversationBufferMemory is implemented
        # For simplicity, we're assuming it's already linked

        # Generate AI response using the QA chain
        result = qa_chain({"question": user_message})
        ai_response = result["answer"]
        logger.debug(f"AI response for session {session_id} with RAG: {ai_response}")

        # Add AI response to memory
        memory.chat_memory.add_ai_message(ai_response)
        logger.debug(f"Added AI message to session {session_id} memory")

        return ChatRagResponse(
            session_id=session_id,
            vectorstore_id=vectorstore_id,
            input=user_message,
            response=ai_response
        )

    except PineconeIndexNotFoundError as e:
        raise e
    except PineconeServiceError as e:
        raise e
    except OpenAIServiceError as e:
        raise e
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        logger.error(f"HTTPException: {he.detail}", exc_info=True)
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error in chat_rag_endpoint for session {session_id}")
        raise HTTPException(status_code=500, detail="Failed to process chat message with RAG")
    
# =======================
# Delete Endpoint
# =======================
@app.delete("/delete-session", response_model=DeleteResponse)
async def delete_session(request: DeleteRequest):
    """
    Deletes a chat session and its associated memory
    """
    session_id = request.session_id
    logger.info(f"Delete request initiated for session: {session_id}")
    
    try:
        # Validate session existence
        if session_id not in sessions:
            logger.warning(f"Delete attempt for non-existent session: {session_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
                headers={"X-Session-Status": "not_found"}
            )

        # Perform deletion
        del sessions[session_id]
        remaining = len(sessions)
        
        logger.info(f"Successfully deleted session: {session_id}")
        logger.debug(f"Remaining active sessions: {remaining}")
        
        return DeleteResponse(
            detail=f"Session '{session_id}' has been deleted successfully.",
            session_id=session_id,
            remaining_sessions=remaining
        )

    except HTTPException as http_exc:
        logger.error(f"Deletion failed for session {session_id}: {http_exc.detail}")
        raise http_exc
        
    except KeyError as ke:
        error_msg = f"Session key error: {str(ke)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during deletion",
            headers={"X-Error-Type": "key_error"}
        )
        
    except Exception as e:
        error_msg = f"Unexpected error deleting session {session_id}: {str(e)}"
        logger.exception(error_msg)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
            headers={"X-Error-Type": "unexpected_error"}
        )

# =======================
# Application Startup
# =======================

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("FastAPI application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("FastAPI application shutdown initiated.")