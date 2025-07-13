from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Optional

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
MAX_TEXT_LENGTH = 12000  # Characters
ALLOWED_FILE_TYPES = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg"
}

# Initialize FastAPI
app = FastAPI(
    title="Virtual Vakil API",
    description="Backend for legal document analysis and chatbot",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper Functions
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

async def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with proper error handling"""
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request failed: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail="Legal analysis service is currently unavailable"
        )

# API Endpoints
@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze legal documents (PDF/Images)"""
    try:
        # Validate file type
        if file.content_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_FILE_TYPES.keys())}"
            )

        # Process file
        file_bytes = await file.read()
        extracted_text = (
            extract_text_from_pdf(file_bytes)
            if file.content_type == "application/pdf"
            else extract_text_from_image(file_bytes)
        )[:MAX_TEXT_LENGTH]

        # Generate analysis prompt
        prompt = f"""Analyze this legal document as an expert Indian lawyer:
1. Provide a concise summary (under 200 words)
2. Identify 3-5 key legal clauses
3. Highlight any unusual or concerning terms
4. Explain implications in simple terms

Document Content:
{extracted_text}"""

        # Get analysis from Gemini
        analysis = await call_gemini_api(prompt)
        return {"summary": analysis}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Please try again."
        )

@app.post("/chat-lawyer")
async def chat_lawyer(
    query: str = Form(...),
    context: Optional[str] = Form("")
):
    """Chat endpoint for legal questions"""
    try:
        # Construct prompt with context
        prompt = f"""You are an expert Indian lawyer. Answer concisely but thoroughly.
        
        Previous Context:
        {context if context else 'No previous context'}
        
        New Question:
        {query}
        
        Provide:
        1. Direct answer
        2. Relevant Indian laws
        3. Suggested actions
        4. Warning about potential pitfalls"""

        # Get response from Gemini
        response = await call_gemini_api(prompt)
        return {"response": response}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Chat service is currently unavailable"
        )

@app.get("/health")
async def health_check():
    """Endpoint for health checks"""
    return {
        "status": "healthy",
        "service": "Virtual Vakil Backend",
        "version": "1.0.0"
    }

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )