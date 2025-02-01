import logging
import time
import traceback  # Ensure traceback is imported
import os
import json
import platform
import traceback
from typing import Optional, Union, List, Any

from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import docx2txt
import fitz  # PyMuPDF for reading PDFs
from pdf2image import convert_from_path
import PyPDF2
import pikepdf
from PIL import Image, ImageEnhance, ImageFilter

from pytesseract import image_to_string, pytesseract
import httpx  # For making internal HTTP requests
import json
import base64
import gc
import Prompts

from groq import Groq# Ensure this import matches Groq's client library
from cerebras.cloud.sdk import Cerebras  # Ensure this SDK is installed

import boto3
import hashlib
import pandas as pd
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import awswrangler as wr

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

UPLOAD_DIR = os.path.abspath("documents")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created upload directory at {UPLOAD_DIR}")
    
# =========================================
# Pydantic Models
# =========================================

class DocumentResponse(BaseModel):
    FileName: Optional[str] = None
    Text: Optional[str] = None

class CombinedDocumentResponse(BaseModel):
    FileName: Optional[str] = None
    Text: Optional[str] = None
    DocumentType: Optional[str] = None
    Content: Optional[dict] = None

# =========================================
# Constants and Configuration
# =========================================

# Pydantic model for parse request
class ParseRequest(BaseModel):
    document_text: str = Field(..., example="Your document text here.")
    system_prompt: Optional[str] = Field(
        None,
        example="You are an assistant that extracts key information from documents."
    )
    temperature: Optional[float] = Field(1.0, example=0.7)
    max_tokens: Optional[int] = Field(2048, example=1500)
    top_p: Optional[float] = Field(0.9, example=0.8)
    stream: Optional[bool] = Field(False, example=True)
    response_format: Optional[dict] = Field({"type": "json_object"}, example={"type": "json_object"})
    stop: Optional[Union[str, List[str]]] = Field(None, example=["\n"])

# Pydantic model for parse response
class ParseResponse(BaseModel):
    status: str
    details: Optional[dict] = None
    

# =============================
# Models
# =============================

class QueryRequest(BaseModel):
    query: str
    query_type: str  # 'neptune' or 'ims'

class QueryResponse(BaseModel):
    query: str
    response: dict  # or use `Any` for more flexibility

# Load API Key from environment
DEFAULT_MODEL = "llama-3.2-90b-vision-preview"  # Default model for OCR

GROQ_API_KEY = os.getenv("GROQ_API_KEY")# Replace with your actual API key or set as environment variable
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")# Replace with your actual API key or set as environment variable

# AWS credentials should be loaded from environment variables
credentials = {
    'AccessKeyId': os.getenv('AWS_ACCESS_KEY_ID'),
    'SecretAccessKey': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'Token': os.getenv('AWS_SESSION_TOKEN'),
    'Expiration': '2025-01-25T21:52:53Z',
    'Code': 'Success',
    'Message': None
}

# Define prompts or other constants if needed
# class Prompts:
#     DOCUMENT_SYSTEM_PROMPT = "You are a helpful assistant that parses documents into structured JSON."
    
# =============================
# Utility Functions
# =============================

def get_account_hash_from_account_id(account_id: str):
    """Convert account ID to a hashed string."""
    return hashlib.md5(account_id.encode("utf-8")).hexdigest()

def create_session_with_credentials(credentials: dict, region: str = "us-east-1"):
    """
    Create a boto3 Session using explicit credentials.
    """
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials.get('Token'),
        region_name=region
    )

def create_session_with_env_credentials(region: str = "us-east-1"):
    """
    Create a boto3 Session using environment variables.
    This will return a session with no credentials if they are not found.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=region
    )

def get_aws_auth_token(session):
    """
    Extract the frozen credentials and other details needed for SigV4 signing.
    """
    creds = session.get_credentials()
    if not creds:
        # No credentials found in session
        raise ValueError("No valid AWS credentials found in session. "
                         "Make sure environment variables are set or fallback credentials are provided.")
    credentials = creds.get_frozen_credentials()
    region = session.region_name or "us-east-1"
    service_name = "execute-api"  # For Neptune's HTTP endpoint
    return {
        "credentials": Credentials(
            credentials.access_key,
            credentials.secret_key,
            credentials.token
        ),
        "service_name": service_name,
        "region": region
    }

def get_account_hash_from_session(session):
    """
    Retrieve the AWS account ID via STS and compute its hash.
    """
    sts_client = session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    return get_account_hash_from_account_id(account_id)

def get_intelligence_base_url_from_session(session):
    """
    Build the base URL for Neptune's SPARQL endpoint.
    """
    account_hash = get_account_hash_from_session(session)
    # Example: https://intelligence.{account_hash}.gryps.io
    return f"https://intelligence.{account_hash}.gryps.io"

# =============================
# Query Handlers
# =============================

# IMS Query Handler
class IMSQueryHandler:
    def __init__(self, session: boto3.Session):
        self.session = session

    def list_of_databases(self):
        """
        List available databases in AWS Glue Catalog.
        """
        try:
            return wr.catalog.databases(boto3_session=self.session)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def list_of_tables(self, database: str):
        """
        List available tables in a given database.
        """
        try:
            return wr.catalog.tables(database=database, boto3_session=self.session)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def query(self, query: str, database: str):
        """
        Execute an Athena query against the specified database.
        """
        try:
            df = wr.athena.read_sql_query(
                query,
                database=database,
                boto3_session=self.session,
                ctas_approach=False,
                workgroup="AmazonAthenaLakeFormation"
            )
            # Return as a list of dictionaries
            return df.to_dict(orient='records')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Neptune Query Handler
class NeptuneQueryHandler:
    def __init__(self, session: boto3.Session):
        auth_details = get_aws_auth_token(session)
        self.credentials = auth_details["credentials"]
        self.service_name = auth_details["service_name"]
        self.region = auth_details["region"]
        self.base_url = get_intelligence_base_url_from_session(session)

    def query(self, query: str, output_format="json"):
        """
        Execute a SPARQL query against Amazon Neptune.
        """
        payload = json.dumps({"query": query})
        headers = {"Content-Type": "application/json"}

        aws_request = AWSRequest(
            method="POST",
            url=f"{self.base_url}/sparql",
            data=payload,
            headers=headers
        )

        # Sign the request using SigV4
        SigV4Auth(self.credentials, self.service_name, self.region).add_auth(aws_request)
        signed_headers = dict(aws_request.headers)

        try:
            response = requests.post(
                aws_request.url,
                headers=signed_headers,
                data=payload,
                timeout=180
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # If response exists, use its status code; otherwise, default to 500
            status_code = response.status_code if 'response' in locals() and response else 500
            raise HTTPException(status_code=status_code, detail=str(e))

        if output_format == "json":
            return response.json()
        elif output_format == "pandas":
            return self._convert_to_pandas(response.json())
        else:
            raise HTTPException(status_code=400, detail="Invalid output format. Use 'json' or 'pandas'.")

    def _convert_to_pandas(self, response: dict):
        """
        Convert Neptune SPARQL JSON response to a list of dictionaries (like Pandas records).
        """
        results = []
        for item in response.get("results", {}).get("bindings", []):
            row = {k: v["value"] for k, v in item.items()}
            results.append(row)
        return results
    
s# =============================
# Initialize Query Handlers
# =============================

# Option 1: Using Temporary Credentials (Not Recommended for Production)
# credentials = {
#     'AccessKeyId': 'YOUR_ACCESS_KEY_ID',
#     'SecretAccessKey': 'YOUR_SECRET_ACCESS_KEY',
#     'Token': 'YOUR_SESSION_TOKEN',
#     'Expiration': 'EXPIRATION_TIME',
#     'Code': 'Success',
#     'Message': None
# }
# session = create_session_with_credentials(credentials)
# neptune_client = NeptuneQueryHandler(session=session)
# ims_client = IMSQueryHandler(session=session)

# Option 2: Using Environment Variables (Recommended)
FALLBACK_CREDENTIALS = {}

# First, try to get a session from environment variables.
session = create_session_with_env_credentials()

# If that didn't work (credentials are missing), fallback to a known set.
if session.get_credentials() is None:
    session = create_session_with_credentials(FALLBACK_CREDENTIALS)

# Initialize Neptune and IMS clients
neptune_client = NeptuneQueryHandler(session=session)
ims_client = IMSQueryHandler(session=session)

# =========================================
# Utility Functions
# =========================================

def ocr_image(image_path):
    """
    Extracts text from an image using pytesseract.
    """
    if platform.system() == "Windows":
        pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        logger.info(f"Starting OCR with pytesseract on image: {image_path}")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        logger.info(f"Succesfully performed pytesseract OCR on image: {image_path}")
        return text
    except Exception as e:
        logger.error(f"Error performing pytesseract OCR: {e} on image: {image_path}")
        return ""

# def extract_text_with_ollama(image_path: str, model: str, temperature: float, max_tokens: int, top_p: float, top_k: int) -> str:
#     logger.info(f"Starting OCR with Ollama on image {image_path} using model: {model}")
#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[
#                 {
#                     'role': 'user',
#                     'content': (Prompts.OCR_SYSTEM_PROMPT.strip()),
#                     'images': [base64.b64encode(open(image_path, "rb").read()).decode('utf-8')]
#                 }
#             ],
#             options={
#                 'temperature': temperature,
#                 'max_tokens': max_tokens,
#                 'top_p': top_p,
#                 'top_k': top_k,
#                 'stream': False,
#             }
#         )
#         ocr_text = response.message.content.strip()
#         logger.info(f"OCR successful for image: {image_path} using model: {model}")
#         logger.info(f"Text Parsed using model {model}: {ocr_text}")
#         # tesseract_text = ocr_image(image_path)
#         # combined_text = ocr_text + "\n" + tesseract_text
#         logger.info(f"OCR successful for image: {image_path} using model: {model} and pytesseract.")
#         return ocr_text
#     except Exception as e:
#         logger.error(f"Error during OCR with Ollama for image {image_path} using model {model}: {e}")
#         return f"Error performing OCR on the image: {e}"

def extract_text_with_ollama(image_path: str, model: str = "llama-3.2-90b-vision-preview",
                          temperature: float = 1.0, max_tokens: int = 1024,
                          top_p: float = 1.0, top_k: int = 0) -> str:
    """
    Extracts text from an image using Groq's OCR model.

    :param image_path: Path to the image file.
    :param model: The Groq model to use for OCR.
    :param temperature: Sampling temperature.
    :param max_tokens: Maximum number of tokens in the response.
    :param top_p: Nucleus sampling parameter.
    :param top_k: Top-K sampling parameter.
    :return: Extracted text or an error message.
    """
    logger.info(f"Starting OCR with Groq on image {image_path} using model: {model}")
    
    try:
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)

        # Convert image to data URL
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_data_url = f"data:image/{image_path.split('.')[-1]};base64,{encoded_image}"

        # Define the prompt
        prompt = [
            {
                "type": "text",
                "text": (
                    "You are an OCR algorithm. Provide the OCR text. Act as Just an OCR engine, "
                    "Nothing more, Just OCR no additional Reasoning.\n"
                    "Please do not add any extra information. Try your best to fetch all the text from the image even if it is not clear.\n"
                    "If the image is scanned or has low quality, try harder to extract all the text from the image.\n"
                    "When images contain handwriting, convert it to text as well.\n"
                    "If text has multiple columns, read from left to right and then top to bottom.\n"
                    "Make sure to properly extract numbers, dates, and special characters.\n"
                    "Do not ask for confirmation. Directly provide the OCR text.\n"
                    "Don't skip any text in the image. Do not hallucinate or make up text. Do not make any spelling mistakes.\n"
                    "Provide the text as it is in the image. Do not add any extra information. Do not ask for confirmation."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
                }
            }
        ]

        # Create the completion
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=None,
        )

        # Extract OCR text from the response
        ocr_text = completion.choices[0].message.content
        logger.info(f"OCR successful for image: {image_path} using model: {model}")
        logger.info(f"Text Parsed using model {model}: {ocr_text}")

        return ocr_text

    except Exception as e:
        logger.error(f"Error during OCR with Groq for image {image_path} using model {model}: {e}")
        return f"Error performing OCR on the image: {e}"

def extract_text_from_pdf_with_ollama(pdf_path: str, poppler_path: Optional[str], model: str, temperature: float, max_tokens: int, top_p: float, top_k: int) -> str:
    logger.info(f"Attempting Ollama-based OCR for scanned PDF: {pdf_path} using model: {model}")
    try:
        pages = convert_from_path(pdf_path, dpi=100, poppler_path=poppler_path)
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return "Error converting PDF to images."

    all_text = []

    for page_num, page in enumerate(pages, 1):
        logger.info(f"Processing page {page_num} of {pdf_path} for OCR using model: {model}")
        image = page.convert("RGB")
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, format="JPEG")
            temp_image_path = temp_file.name

        ocr_text = extract_text_with_ollama(
            image_path=temp_image_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        all_text.append(f"Page {page_num}:\n{ocr_text}\n")
        gc.collect()

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logger.info(f"Temporary image file {temp_image_path} deleted.")

    return "\n".join(all_text)

def read_pdf(file_path: str, dpi: int, model: str, temperature: float, max_tokens: int, top_p: float, top_k: int, repair_attempted: bool = False) -> Optional[str]:
    logger.info(f"Reading PDF file: {file_path} with model: {model}")
    text = ""

    poppler_path = None
    if platform.system() == "Windows":
        poppler_path = r"C:\Users\karan\Documents\Projects\poppler-24.08.0\Library\bin"

    # Attempt direct extraction via PyMuPDF
    try:
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page_text = pdf[page_num].get_text()
                text += page_text
            
            # ocr_text = extract_text_from_pdf_with_ollama(
            #     pdf_path=file_path,
            #     poppler_path=poppler_path,
            #     model=model,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     top_p=top_p,
            #     top_k=top_k
            # )
        if text.strip():
            logger.info(f"Successfully extracted text using PyMuPDF from {file_path}")
            # return text + "\n" + ocr_text
            return text
    except Exception as e:
        logger.error(f"Error reading PDF with PyMuPDF: {e}")

    # If direct extraction fails, use Ollama-based OCR
    logger.info(f"OCR extraction with Ollama on scanned PDF: {file_path}")
    ocr_text = extract_text_from_pdf_with_ollama(
        pdf_path=file_path,
        poppler_path=poppler_path,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k
    )
    if ocr_text.strip() and not ocr_text.startswith("Error performing OCR"):
        logger.info(f"Successfully extracted text using Ollama OCR from {file_path}")
        return ocr_text
    else:
        # If OCR failed, attempt fallback to default model if not already default
        if model != DEFAULT_MODEL:
            logger.warning(f"OCR failed with model {model}, attempting fallback with default model {DEFAULT_MODEL}")
            fallback_ocr_text = extract_text_from_pdf_with_ollama(
                pdf_path=file_path,
                poppler_path=poppler_path,
                model=DEFAULT_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            if fallback_ocr_text.strip() and not fallback_ocr_text.startswith("Error performing OCR"):
                logger.info(f"OCR succeeded using fallback model {DEFAULT_MODEL} for {file_path}")
                return fallback_ocr_text
            else:
                logger.error(f"Fallback model {DEFAULT_MODEL} also failed for {file_path}")

    # If OCR fails, try repairing the PDF
    if not repair_attempted:
        try:
            with pikepdf.open(file_path) as pdf:
                repaired_pdf = pikepdf.new()
                repaired_pdf.pages.extend(pdf.pages)
                repaired_pdf.save(file_path)
            logger.info(f"Repaired PDF: {file_path}. Retrying extraction.")
            return read_pdf(
                file_path=file_path,
                dpi=dpi,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                repair_attempted=True
            )
        except Exception as e:
            logger.error(f"Error repairing PDF: {e}")

    # As a last resort, try PyPDF2
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        if text.strip():
            logger.info(f"Successfully extracted text using PyPDF2 from {file_path}")
            return text
    except Exception as e:
        logger.error(f"Error reading with PyPDF2: {e}")

    logger.error(f"Unable to extract text from the PDF: {file_path}")
    return None

def read_txt(file_path: str) -> Optional[str]:
    logger.info(f"Reading TXT file: {file_path}")
    encodings = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                text = f.read()
            logger.info(f"Successfully read TXT file with encoding: {enc}")
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            return None
    logger.error(f"Failed to decode TXT file with known encodings: {file_path}")
    return None

def read_docx(file_path: str) -> Optional[str]:
    logger.info(f"Reading DOCX file: {file_path}")
    try:
        text = docx2txt.process(file_path)
        logger.info("Successfully read DOCX file using docx2txt.")
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX file: {e}")
        return None

def read_image(file_path: str, model: str, temperature: float, max_tokens: int, top_p: float, top_k: int) -> Optional[str]:
    """
    Reads text from an image file using OCR via Ollama with optional parameter tweaks.
    """
    logger.info(f"Reading Image file: {file_path} with Ollama OCR model: {model}")
    try:
        image = Image.open(file_path).convert("RGB")
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, format="JPEG")
            temp_image_path = temp_file.name

        # Perform OCR with Ollama with specified parameters
        ocr_text = extract_text_with_ollama(
            image_path=temp_image_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )
        if ocr_text.strip() and not ocr_text.startswith("Error performing OCR"):
            logger.info(f"Successfully performed OCR on image file {file_path} using {model}")
            os.remove(temp_image_path)
            return ocr_text
        else:
            # Attempt fallback if not default model
            if model != DEFAULT_MODEL:
                logger.warning(f"OCR failed with model {model} on image {file_path}, attempting fallback with default model {DEFAULT_MODEL}")
                fallback_text = extract_text_with_ollama(
                    image_path=temp_image_path,
                    model=DEFAULT_MODEL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k
                )
                if fallback_text.strip() and not fallback_text.startswith("Error performing OCR"):
                    logger.info(f"OCR succeeded on image {file_path} using fallback model {DEFAULT_MODEL}")
                    os.remove(temp_image_path)
                    return fallback_text
                else:
                    logger.error(f"Fallback model {DEFAULT_MODEL} also failed for image {file_path}")
            os.remove(temp_image_path)
            return None
    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {e}")
        return None

def read_document(file_path: str, dpi: int, model: str, temperature: float, max_tokens: int, top_p: float, top_k: int) -> Optional[str]:
    """
    Determines the document type and reads text accordingly.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    logger.info(f"Determining document type for {file_path}, using model: {model}")

    try:
        if ext == '.txt':
            return read_txt(file_path)
        elif ext == '.docx':
            return read_docx(file_path)
        elif ext == '.pdf':
            return read_pdf(
                file_path=file_path,
                dpi=dpi,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return read_image(
                file_path=file_path,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        logger.error(f"Error reading document {file_path}: {e}")
        return None

def parse_document(file_path: str, dpi: int = 300, model: str = DEFAULT_MODEL, temperature: float = 0.1, max_tokens: int = 4096, top_p: float = 0.9, top_k: int = 50) -> Optional[str]:
    """
    Parses the document and returns extracted text.
    """
    logger.info(f"Parsing document: {file_path} with model: {model}")
    extracted_text = read_document(
        file_path=file_path,
        dpi=dpi,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k
    )
    if extracted_text:
        logger.info("Document parsing successful.")
        return extracted_text
    else:
        logger.error("Failed to extract text from the document.")
        return None

def parse_text_with_groq(document_text: str, api_key: str) -> Optional[dict]:
    """
    Parses the extracted text using Groq's LLM and returns structured JSON.
    """
    logger.info("Starting parsing of extracted text with Groq's LLM.")
    try:
        # client = Groq(api_key=api_key)  # Initialize the Groq client with the API key
        # Call the Groq chat completion API
        client = Cerebras(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {
                    "role": "system",
                    "content": Prompts.DOCUMENT_SYSTEM_PROMPT.strip()
                },
                {
                    "role": "user",
                    "content": document_text
                }
            ],
            temperature=1,
            max_tokens=8192,
            top_p=1,
            stream=False,
            response_format={"type":"json_object"},
            stop=None,
        )

        # Gather and return the output
        parsed_content = completion.choices[0].message.content
        try:
            JSONOutput = json.loads(parsed_content)
            logger.info("Successfully parsed text with Groq's LLM.")
            return JSONOutput
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Groq's response: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating system prompt for Groq's LLM: {e}") # Log the error 
        return None

def extract_text_from_document(file_path: str, dpi: int, model: str) -> Optional[dict]:
    """
    Extracts text from a document and returns a dictionary with FileName and Text.
    """
    logger.info(f"Extracting text from document: {file_path}")
    extracted_text = parse_document(file_path, dpi=dpi, model=model)
    if extracted_text:
        return {"FileName": os.path.basename(file_path), "Text": extracted_text}
    else:
        return None

app = FastAPI(root_path="/api")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources.")

# Application Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Releasing resources.")

# Middleware to log request and response details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request received from {client_host}: {request.method} {request.url}")

    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        raise

    process_time = time.time() - start_time
    response_log = f"Response: {response.status_code} | Processing time: {process_time:.2f}s"

    if response.status_code >= 500:
        logger.error(response_log)
    elif response.status_code >= 400:
        logger.warning(response_log)
    else:
        logger.info(response_log)

    return response

# Global exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

# Global exception handler
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc} - Path: {request.url.path}"
    )
    logger.debug(traceback.format_exc())  # Now, traceback is defined
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

# Health check endpoint with detailed logging
@app.get("/health")
async def health_check():
    logger.info("Starting health check")
    try:
        # Add actual health checks here (e.g., database connection)
        logger.debug("Performing health check validations")
        return {"status": "healthy", "details": "All systems operational"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Service unavailable",
        ) from e
    finally:
        logger.info("Completed health check")

def LLM_Text_Parse(
    document_text: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    top_p: float = 1.0,
    stream: bool = False,
    response_format: dict = {"type": "json_object"},
    stop: Optional[Union[str, List[str]]] = None
) -> Optional[dict]:
    """
    Parses the extracted text using Cerebras' LLM and returns structured JSON.
    """
    logger.info("Starting parsing of extracted text with Cerebras' LLM.")
    try:
        client = Cerebras(api_key=api_key)
        
        # Use the provided system prompt or default if not provided
        prompt = system_prompt if system_prompt else Prompts.DOCUMENT_SYSTEM_PROMPT.strip()
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": document_text
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            response_format=response_format,
            stop=stop,
        )

        # Gather and return the output
        parsed_content = completion.choices[0].message.content
        try:
            JSONOutput = json.loads(parsed_content)
            logger.info("Successfully parsed text with Cerebras' LLM.")
            return JSONOutput
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Cerebras' response: {e}")
            return None

    except Exception as e:
        logger.error(f"Error generating system prompt for Cerebras' LLM: {e}", exc_info=True)
        return None

# New endpoint to parse text using LLM with customizable parameters
@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    logger.info("Received request to parse document.")
    logger.debug(f"Document text received: {request.document_text[:100]}...")  # Log first 100 chars

    try:
        parsed_result = LLM_Text_Parse(
            document_text=request.document_text,
            api_key=CEREBRAS_API_KEY,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream,
            response_format=request.response_format,
            stop=request.stop
        )
        if parsed_result is None:
            logger.error("Parsing failed due to invalid JSON response.")
            raise HTTPException(status_code=500, detail="Failed to parse document.")

        logger.info("Document parsed successfully.")
        return ParseResponse(status="success", details=parsed_result)

    except HTTPException as he:
        logger.warning(f"HTTPException during parsing: {he.detail}")
        raise he  # Re-raise to be handled by global handlers

    except Exception as e:
        logger.error(f"Unexpected error during document parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during parsing.")
    
# @app.post("/query", response_model=QueryResponseModel)
# def execute_query(request: QueryRequestModel):
#     logger.info("Received Neptune query request.")
#     logger.debug(f"Neptune query received: {request.query}")

#     try:
#         data = neptune_client.query(request.query, output_format="json")
#         logger.info("Neptune query executed successfully.")
#         return QueryResponseModel(query=request.query, response=data)

#     except HTTPException as he:
#         # Re-raise HTTPExceptions to be handled by the global exception handler
#         logger.warning(f"HTTPException during Neptune query: {he.detail}")
#         raise he

#     except Exception as e:
#         # Log unexpected exceptions and raise a generic HTTPException
#         logger.error(f"Unexpected error during Neptune query: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error during Neptune query.")

# if __name__ == "__main__":
#     uvicorn.run("api2:app", host="0.0.0.0", port=8000)

@app.post("/execute-query", response_model=QueryResponse)
def execute_query(query_request: QueryRequest):
    """
    Execute a query against AWS Neptune (SPARQL) or AWS IMS (Athena) 
    and return the results.
    """
    query_type = query_request.query_type.lower()
    query = query_request.query

    if query_type == "neptune":
        try:
            response = neptune_client.query(query, output_format="json")
            return QueryResponse(query=query, response=response)
        except HTTPException as http_exc:
            # Re-raise HTTP exceptions as-is
            raise http_exc
        except Exception as e:
            # Catch any unexpected errors
            raise HTTPException(status_code=500, detail=str(e))

    elif query_type == "ims":
        # In this example, we assume a default Athena database
        default_database = "your_default_database"
        try:
            response = ims_client.query(query, database=default_database)
            return QueryResponse(query=query, response=response)
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400, 
            detail="Invalid query type. Use 'neptune' or 'ims'."
        )

@app.get("/list-databases", response_model=dict)
def list_databases():
    """
    List available databases from the AWS Glue Catalog (for IMS).
    """
    try:
        databases = ims_client.list_of_databases()
        return {"databases": databases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-tables/{database}", response_model=dict)
def list_tables(database: str):
    """
    List tables in a specified database from the AWS Glue Catalog (for IMS).
    """
    try:
        tables = ims_client.list_of_tables(database)
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# =========================================
# API Endpoints
# =========================================

@app.post("/extract_text", response_model=DocumentResponse)
async def extract_text(
    file: UploadFile = File(...),
    model: str = DEFAULT_MODEL,
    temperature: float = Query(0.1, ge=0.0, le=1.0, description="Sampling temperature"),
    max_tokens: int = Query(4096, ge=1, description="Maximum number of tokens"),
    top_p: float = Query(0.7, ge=0.0, le=1.0, description="Nucleus sampling parameter"),
    top_k: int = Query(50, ge=1, description="Top-K sampling parameter")
    ):
    # Validate the model parameter
    # If the model is invalid or fails, we will attempt fallback inside the OCR functions.
    logger.info(f"extract_text API called with model: {model}")

    filename = file.filename
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    supported_extensions = ['.txt', '.docx', '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    if ext not in supported_extensions:
        error_msg = f"Unsupported file type: {ext}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Generate a unique filename to prevent collisions
    unique_id = uuid.uuid4().hex
    unique_filename = f"{unique_id}_{filename}"
    temp_file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Stream the file in chunks to reduce memory usage
        logger.info(f"Starting to stream the uploaded file: {unique_filename}")
        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(1024*1024)  # Read in 1 MB chunks
                if not chunk:
                    break
                temp_file.write(chunk)
        logger.info(f"Uploaded file saved to {temp_file_path}")

        extracted_text = parse_document(
            file_path=temp_file_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )

    except Exception as e:
        error_msg = f"Failed to process the file: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail="Failed to extract text from the document.")

    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted.")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {e}")

    if extracted_text:
        if not isinstance(extracted_text, str):
            logger.error("Extracted text is not a string.")
            raise HTTPException(status_code=500, detail="Extracted text is invalid.")
        return DocumentResponse(FileName=filename, Text=extracted_text)
    else:
        error_msg = "Failed to extract text from the document."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# =========================================
# New Endpoint: /process_document
# =========================================

@app.post("/process_document", response_model=CombinedDocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    model: str = DEFAULT_MODEL,
    temperature: float = Query(0.1, ge=0.0, le=1.0, description="Sampling temperature"),
    max_tokens: int = Query(4096, ge=1, description="Maximum number of tokens"),
    top_p: float = Query(0.7, ge=0.0, le=1.0, description="Nucleus sampling parameter"),
    top_k: int = Query(50, ge=1, description="Top-K sampling parameter")
    ):
    """
    API endpoint that extracts text from a document, parses it using Groq's LLM, and returns a combined JSON response.
    """
    logger.info(f"process_document API called with model: {model}")

    filename = file.filename
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    supported_extensions = ['.txt', '.docx', '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    if ext not in supported_extensions:
        error_msg = f"Unsupported file type: {ext}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Generate a unique filename to prevent collisions
    unique_id = uuid.uuid4().hex
    unique_filename = f"{unique_id}_{filename}"
    temp_file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Stream the file in chunks to reduce memory usage
        logger.info(f"Starting to stream the uploaded file: {unique_filename}")
        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(1024*1024)  # Read in 1 MB chunks
                if not chunk:
                    break
                temp_file.write(chunk)
        logger.info(f"Uploaded file saved to {temp_file_path}")

        # Step 1: Extract text from the document by calling /extract_text internally
        # Using httpx to make an internal HTTP request
        async with httpx.AsyncClient(timeout=httpx.Timeout(3000.0)) as client:
            with open(temp_file_path, "rb") as f:
                files = {'file': (unique_filename, f, file.content_type)}
                # Assuming the FastAPI app is running on localhost:8000
                # Adjust the URL if running on a different host/port
                extract_response = await client.post(
                    "http://127.0.0.1:8000/extract_text",
                    files=files,
                    params={"model": model, "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "top_k": top_k}
                )

        if extract_response.status_code != 200:
            error_msg = f"Failed to extract text from the document. Status Code: {extract_response.status_code}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        extract_data = extract_response.json()
        file_name = extract_data.get("FileName")
        document_text = extract_data.get("Text")

        if not document_text:
            error_msg = "Extracted text is empty."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Step 2: Parse the extracted text using Groq's API
        parsed_output = parse_text_with_groq(document_text, api_key=CEREBRAS_API_KEY)

        if not parsed_output:
            error_msg = "Failed to parse extracted text with Groq's LLM."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Step 3: Combine all information into the response
        combined_response = CombinedDocumentResponse(
            DocumentType=parsed_output.get("DocumentType"),
            Content=parsed_output.get("Content"),
            FileName=file_name,
            Text=document_text
        )

        return combined_response

    except HTTPException as he:
        raise he  # Re-raise HTTPExceptions to be handled by FastAPI
    except Exception as e:
        error_msg = f"Failed to process the document: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail="Failed to process the document.")

    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted.")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {e}")