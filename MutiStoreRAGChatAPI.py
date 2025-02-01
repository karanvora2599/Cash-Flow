import os
import uuid
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import List, Dict, Any, Optional, Set

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import BaseRetriever, Document

from cerebras.cloud.sdk import Cerebras
import pinecone
from dotenv import load_dotenv
import openai  # Ensure openai is installed
from threading import Lock

# =======================
# Logging Configuration
# =======================
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

# =======================
# Load Environment Variables
# =======================
load_dotenv()

# =======================
# Configuration
# =======================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Optional if using Groq
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define your Pinecone environment (e.g., 'us-west1-gcp')
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

# Index configurations
INDEX_CONFIG = {
    "legal-index": {"name": "legal-docs", "top_k": 2},
    "tech-index": {"name": "technical-docs", "top_k": 3}
}

# System prompt (you can customize this as needed)
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant that helps with productivity tips."

# =======================
# Custom Exceptions
# =======================
class PineconeIndexNotFoundError(Exception):
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.message = f"Pinecone index '{self.index_name}' does not exist."
        super().__init__(self.message)

class PineconeServiceError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class OpenAIServiceError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# =======================
# Pydantic Models
# =======================

# Session Models
class SessionStartRequest(BaseModel):
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT  # Optional system prompt

class SessionStartResponse(BaseModel):
    session_id: str

class DeleteRequest(BaseModel):
    session_id: str

class DeleteResponse(BaseModel):
    detail: str
    session_id: str
    remaining_sessions: int

# Chat Models
class ChatRagRequest(BaseModel):
    session_id: str
    vector_stores: List[str]
    message: str
    system_prompt: Optional[str] = None  # Optional system prompt per chat

class ChatRagResponse(BaseModel):
    session_id: str
    vector_stores: List[str]
    input: str
    response: str

# =======================
# Client Initialization
# =======================

# Validate environment variables
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable not set.")
    raise EnvironmentError("PINECONE_API_KEY environment variable not set")
if not CEREBRAS_API_KEY:
    logger.error("CEREBRAS_API_KEY environment variable not set.")
    raise EnvironmentError("CEREBRAS_API_KEY environment variable not set")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

try:
    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    pc = pinecone.Index  # Pinecone client doesn't need instantiation
    logger.info(f"Pinecone initialized with environment: {PINECONE_ENVIRONMENT}")
except Exception as e:
    logger.exception("Failed to initialize Pinecone.")
    raise EnvironmentError(f"Failed to initialize Pinecone: {e}")

try:
    # Initialize Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    logger.info("OpenAI Embeddings initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize OpenAI Embeddings.")
    raise EnvironmentError(f"Failed to initialize OpenAI Embeddings: {e}")

try:
    # Initialize Cerebras Client
    cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
    logger.info("Cerebras client initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize Cerebras client.")
    raise EnvironmentError(f"Failed to initialize Cerebras client: {e}")

# =======================
# FastAPI Initialization
# =======================
app = FastAPI(title="Multi-Vectorstore RAG Chat API with Sessions")

# =======================
# In-Memory Session Storage
# =======================

# Note: For production, replace with a persistent database like Redis
sessions: Dict[str, Dict[str, Any]] = {}

# Lock for thread-safe session management
session_lock = Lock()

# =======================
# Utility Functions
# =======================

def retry_operation(operation, retries=3, delay=2):
    """
    Retry an operation multiple times with delay.
    """
    for attempt in range(retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for operation: {operation.__name__}")
            return operation()
        except (pinecone.PineconeException, openai.error.OpenAIError) as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            logger.exception(f"Unexpected error during operation '{operation.__name__}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error during operation: {e}"
            )
    logger.error(f"All {retries} attempts failed for operation '{operation.__name__}'.")
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="External service is unavailable. Please try again later."
    )

def load_vectorstore(vectorstore_id: str):
    """
    Load a Pinecone vector store based on the vectorstore_id.
    """
    try:
        if vectorstore_id not in pinecone.list_indexes():
            logger.error(f"Pinecone index '{vectorstore_id}' does not exist.")
            raise PineconeIndexNotFoundError(vectorstore_id)
        
        vector_store = LangchainPinecone.from_existing_index(
            index_name=vectorstore_id,
            embedding=embeddings,
            client=pc  # Pass the Pinecone client instance
        )
        logger.info(f"Pinecone vector store '{vectorstore_id}' loaded successfully.")
        return vector_store
    except PineconeIndexNotFoundError as e:
        logger.error(e.message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
    except PineconeServiceError as e:
        logger.error(e.message)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=e.message)
    except Exception as e:
        logger.exception(f"Unexpected error while loading vector store '{vectorstore_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load vector store.")

def create_qa_chain(vector_store: LangchainPinecone, memory: ConversationBufferMemory):
    """
    Create a ConversationalRetrievalChain with the provided vector store and memory.
    """
    try:
        # Initialize RAG pipeline
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=CerebrasLLMWrapper(client=cerebras_client, model="llama3.1-8b"),
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

class CerebrasLLMWrapper:
    def __init__(self, client: Cerebras, model_name: str = "llama3.1-8b"):
        self.client = client
        self.model = model_name

    def __call__(self, prompt: str) -> str:
        try:
            logger.debug(f"Sending prompt to Cerebras LLM: {prompt}")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3
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
# API Endpoints
# =======================
    
@app.post("/start-session", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Initializes a new chat session with a unique ID.
    """
    logger.info("Received request to start a new session.")
    try:
        session_id = str(uuid.uuid4())
        memory = ConversationBufferMemory()
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        memory.chat_memory.add_system_message(system_prompt)
        with session_lock:
            sessions[session_id] = {
                "memory": memory,
                "qa_chains": {}  # To store QA chains per vector store
            }
        logger.info(f"Started new session with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        logger.exception("Error starting new session.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start session.")

@app.delete("/delete-session", response_model=DeleteResponse)
async def delete_session(request: DeleteRequest):
    """
    Deletes a chat session and its associated memory.
    """
    session_id = request.session_id
    logger.info(f"Received request to delete session: {session_id}")
    
    try:
        with session_lock:
            if session_id not in sessions:
                logger.warning(f"Delete attempt for non-existent session: {session_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found",
                    headers={"X-Session-Status": "not_found"}
                )
            del sessions[session_id]
            remaining = len(sessions)
        logger.info(f"Successfully deleted session: {session_id}. Remaining sessions: {remaining}")
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during deletion",
            headers={"X-Error-Type": "key_error"}
        )
    except Exception as e:
        error_msg = f"Unexpected error deleting session {session_id}: {e}"
        logger.exception(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
            headers={"X-Error-Type": "unexpected_error"}
        )

@app.post("/chat", response_model=ChatRagResponse)
async def chat_endpoint(chat_request: ChatRagRequest):
    """
    Handles chat messages for a specific session. Allows an optional system prompt per message.
    """
    session_id = chat_request.session_id
    vector_stores = chat_request.vector_stores
    user_prompt = chat_request.message
    system_prompt = chat_request.system_prompt  # Optional system prompt per chat

    logger.info(f"Received chat message for session: {session_id}")

    # Validate vector stores
    invalid_stores = [store for store in vector_stores if store not in INDEX_CONFIG]
    if invalid_stores:
        logger.error(f"Invalid vector stores requested: {invalid_stores}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid vector stores: {invalid_stores}")

    # Retrieve session
    with session_lock:
        if session_id not in sessions:
            logger.error(f"Session {session_id} not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")
        session = sessions[session_id]
        memory: ConversationBufferMemory = session["memory"]
        qa_chains: Dict[str, ConversationalRetrievalChain] = session.get("qa_chains", {})

    # Load or create QA chains for each vector store
    for store_id in vector_stores:
        if store_id not in qa_chains:
            logger.debug(f"QA chain for vector store '{store_id}' not found. Creating a new one.")
            try:
                vector_store = retry_operation(lambda: load_vectorstore(INDEX_CONFIG[store_id]["name"]))
                qa_chain = create_qa_chain(vector_store, memory)
                qa_chains[store_id] = qa_chain
                logger.info(f"QA chain created for vector store: {store_id}")
            except HTTPException as http_exc:
                logger.error(f"Failed to create QA chain for vector store: {store_id} - {http_exc.detail}")
                raise http_exc
            except Exception as e:
                logger.exception(f"Failed to create QA chain for vector store: {store_id}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create QA chain for vector store: {store_id}")

    # Update the session's QA chains
    with session_lock:
        session["qa_chains"] = qa_chains

    # Prepare the prompt
    if system_prompt:
        combined_prompt = f"{system_prompt}\n{user_prompt}"
        logger.debug("System prompt provided. Prepending to user message.")
    else:
        combined_prompt = user_prompt

    # Aggregate responses from all QA chains
    aggregated_answer = ""
    aggregated_sources: Set[str] = set()

    for store_id, qa_chain in qa_chains.items():
        logger.debug(f"Processing QA chain for vector store: {store_id}")
        try:
            result = qa_chain({"question": combined_prompt})
            answer = result.get("answer", "").strip()
            if answer:
                aggregated_answer += answer + "\n"
                logger.debug(f"Received answer from '{store_id}': {answer}")
            else:
                logger.warning(f"No answer received from '{store_id}' for the prompt.")
            # Collect unique sources
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source_index", store_id)
                if source:
                    aggregated_sources.add(source)
        except HTTPException as http_exc:
            logger.error(f"Error processing QA chain for vector store: {store_id} - {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.exception(f"Error processing QA chain for vector store: {store_id}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing QA chain for vector store: {store_id}")

    # Format the aggregated response
    if aggregated_answer:
        response_text = f"Answer: {aggregated_answer.strip()}\n\nSources:"
    else:
        response_text = "Answer: I'm sorry, I couldn't find an answer to your question.\n\nSources:"

    for source in aggregated_sources:
        response_text += f"\n- {source}"

    # Update conversation memory
    try:
        with session_lock:
            memory.add_user_message(user_prompt)
            if aggregated_answer:
                memory.add_ai_message(aggregated_answer.strip())
            else:
                memory.add_ai_message("I'm sorry, I couldn't find an answer to your question.")
        logger.info(f"Updated conversation memory for session: {session_id}")
    except Exception as e:
        logger.exception(f"Failed to update conversation memory for session: {session_id} - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update conversation memory.")

    logger.info(f"Chat response generated for session: {session_id}")

    return ChatRagResponse(
        session_id=session_id,
        vector_stores=vector_stores,
        input=user_prompt,
        response=response_text
    )

# =======================
# Run the Application
# =======================

# To run the application, use the following command:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload