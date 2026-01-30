"""
PageInsighter API - Optimized FastAPI Application
Text extraction, summarization, translation, and image generation from documents
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import numpy as np
import google.generativeai as genai
from PIL import Image
import PyPDF2
import io
import base64
from dotenv import load_dotenv
import traceback
import json

# Import custom modules
from otherImgGen import ImageGenerator as IG
from image_to_text import ImageToText, explain_image_single_call

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PageInsighter API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
API_KEY = os.getenv("gemini_key")
GEMINI_MODEL = os.getenv("gemini_model", "gemini-pro")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "hindi"
    style: Optional[str] = "realistic"  # FIX: Added style parameter


class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = "hindi"
    explain: bool = False


class TranslateRequest(BaseModel):
    text: str
    language: str = "hindi"


class ImagePromptRequest(BaseModel):
    text: str


# ============================================================================
# ERROR HANDLING
# ============================================================================

def error_response(message: str, status_code: int = 500) -> JSONResponse:
    """Generate consistent error responses"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message
        }
    )


def handle_gemini_error(e: Exception):
    """Centralized error handling for Gemini API"""
    error_msg = str(e).lower()
    
    if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
        raise HTTPException(
            status_code=429,
            detail="API quota exceeded. Please try again later or check your Gemini API limits."
        )
    elif "401" in error_msg or "unauthorized" in error_msg:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your Gemini API configuration."
        )
    elif "400" in error_msg:
        raise HTTPException(
            status_code=400,
            detail="Invalid request to Gemini API. Please try rephrasing your query."
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="AI service temporarily unavailable. Please try again."
        )


# ============================================================================
# FILE UTILITIES
# ============================================================================

def is_image_file(file_content: bytes) -> bool:
    """Check if file is a valid image"""
    try:
        img = Image.open(io.BytesIO(file_content))
        img.verify()
        return True
    except:
        return False


def read_pdf(file_content: bytes) -> str:
    """Extract text from PDF"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text.strip()
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract text from PDF. File may be corrupted."
        )


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Image conversion error: {str(e)}")
        raise


def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    try:
        # Try to open and verify it's a valid image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Bytes conversion error: {str(e)}")
        # Fallback: just encode the bytes directly
        return base64.b64encode(image_bytes).decode('utf-8')


# ============================================================================
# GEMINI API FUNCTIONS
# ============================================================================

def call_gemini_api(prompt: str, image: Optional[Image.Image] = None) -> str:
    """Unified Gemini API call with error handling"""
    try:
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("No response from Gemini API")
        
        return response.text.strip()
    
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        handle_gemini_error(e)


def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using Gemini Vision"""
    prompt = """
    Extract all text from this image exactly as it appears.
    Do not summarize or explain.
    Preserve line breaks, symbols, and formatting as much as possible.
    """
    
    text = call_gemini_api(prompt, image)
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="No text detected in image"
        )
    
    return text


def generate_answer(query: str) -> str:
    """Generate answer using Gemini"""
    prompt = f"""Please provide a detailed response to the following query in HTML format: {query}. 
The response should use proper inline CSS styles for formatting and be well-structured. 

Ensure the HTML is visually appealing and easy to read. 

Additionally, implement the following dynamic styling for readability:
- Font color: dark (#000000)
- Background color: light (#FFFFFF or #F0F0F0)
- Border-radius: 8px
- Padding: 16px

This ensures that the text is always readable regardless of the background color."""
    
    return call_gemini_api(prompt)


def process_text(text: str, explain: bool = False) -> str:
    """Summarize or explain text"""
    if explain:
        prompt = f"""Determine if the input is a mathematical problem or descriptive text. 
- If it is a mathematical problem, explain the steps clearly, provide detailed calculations, and include the final answer. 
- If it is descriptive text, provide a comprehensive explanation with detailed analysis.

Write everything in HTML format with inline styles for rendering. 
Ensure proper formatting for clarity and readability, using appropriate headings, paragraphs, and lists where necessary. 
The response should be visually appealing, with attention to font size, color, and layout to ensure ease of understanding based on the type of input. 

Additionally, implement the following dynamic styling for readability:
- Font color: dark (#000000)
- Background color: light (#FFFFFF or #F0F0F0)
- Border-radius: 8px
- Padding: 16px

The input to be analyzed is: {text}"""
    else:
        prompt = f"Summarize the following text concisely, focusing on key points and main ideas:\n\n{text}"
    
    return call_gemini_api(prompt)


def translate_text(text: str, language: str) -> str:
    """Translate text to specified language"""
    prompt = f"Translate the following text into {language} using simple, clear words:\n\n{text}"
    return call_gemini_api(prompt)


def generate_image_prompts(text: str, count: int = 4) -> List[str]:
    """Generate multiple image generation prompts"""
    prompt = f"""Generate {count} diverse, detailed image generation prompts based on the following text.
Each prompt should:
- Be visually descriptive and unique
- Capture different aspects or key concepts from the content
- Be optimized for 3D ultra HD vibrant image generation
- Be 15-25 words long
- Focus on visualizing the main ideas, not just decorative images

Return ONLY a JSON array of strings, no additional text or markdown.

Text: {text}"""
    
    response = call_gemini_api(prompt)
    
    # Clean response
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    
    try:
        prompts = json.loads(cleaned)
        if isinstance(prompts, list):
            return prompts[:count]
        else:
            raise ValueError("Response is not a list")
    except Exception as e:
        print(f"JSON parsing error: {e}, attempting fallback")
        # Fallback: generate one prompt
        single_prompt = f"A detailed, vibrant 3D visualization representing: {text[:200]}"
        return [single_prompt]


# ============================================================================
# IMAGE GENERATION
# ============================================================================

async def generate_images_from_prompts(prompts: List[str], style: str = "realistic") -> List[dict]:
    """Generate images from prompts with specified style"""
    images = []
    print(f"\nGenerating {len(prompts)} images with style: {style}...")
    
    for i, base_prompt in enumerate(prompts):
        try:
            # FIX: Include the style in the enhanced prompt
            enhanced_prompt = f"3D ultra HD vibrant, {style}, Fantasy and academic {base_prompt}"
            print(f"  [{i+1}/{len(prompts)}] Generating: {enhanced_prompt[:60]}...")
            
            img_bytes, flag = IG(enhanced_prompt)
            
            if flag and img_bytes:
                # Convert bytes to base64
                img_base64 = bytes_to_base64(img_bytes)
                
                images.append({
                    "prompt": base_prompt,
                    "image": img_base64,
                    "index": i + 1
                })
                print(f"  ✓ Image {i+1} generated successfully")
            else:
                print(f"  ✗ Image {i+1} generation failed")
                
        except Exception as img_error:
            print(f"  ✗ Image {i+1} error: {img_error}")
            # Continue with other images
    
    print(f"Successfully generated {len(images)}/{len(prompts)} images\n")
    return images

def answer_query_single_call(query: str, language: str | None):
    model = genai.GenerativeModel(GEMINI_MODEL)

    lang = (language or "english").lower()

    # 🚫 No translation case
    if lang in ["none", "no", "no translation", "english"]:
        prompt = f"""
You are an intelligent tutor.

TASK:
Answer the user's question accurately and clearly.

CONTENT RULES:
- If the question is mathematical or logical → explain step-by-step and give final answer
- If conceptual or theoretical → explain clearly with structure
- If short factual → answer concisely

OUTPUT FORMAT (STRICT):
- Output MUST be valid HTML ONLY
- Use inline CSS
- Background: #F5F5F5
- Text color: #000000
- Border-radius: 8px
- Padding: 16px
- Use:
  <h2>, <p>, <ul>, <li>, <b>

DO NOT:
- Add markdown
- Add text outside HTML
- Mention translation

ONLY RETURN HTML.

User Question:
{query}
"""
    else:
        # 🌍 Translation required
        prompt = f"""
You are an intelligent tutor.

TASK:
Answer the user's question accurately and clearly.

LANGUAGE:
- Translate the FINAL answer into {language}
- Do NOT mention translation explicitly
- Use simple and natural language

CONTENT RULES:
- If the question is mathematical or logical → explain step-by-step and give final answer
- If conceptual → explain clearly and structurally

OUTPUT FORMAT (STRICT):
- Output MUST be valid HTML ONLY
- Use inline CSS
- Background: #F5F5F5
- Text color: #000000
- Border-radius: 8px
- Padding: 16px
- Use:
  <h2>, <p>, <ul>, <li>, <b>

DO NOT:
- Add markdown
- Add explanations outside HTML

ONLY RETURN HTML.

User Question:
{query}
"""

    response = model.generate_content(prompt)

    if not response or not response.text:
        raise Exception("Empty response from Gemini")

    return response.text.strip()




# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PageInsighter API is running",
        "version": "2.0.0",
        "status": "healthy",
        "endpoints": {
            "query": "/api/query",
            "ocr": "/api/ocr",
            "summarize": "/api/summarize",
            "translate": "/api/translate",
            "visualize": "/api/visualize",
            "process_file": "/api/process-file",
            "explain_image": "/api/explain-image"
        }
    }


@app.post("/api/query")
async def answer_query(request: QueryRequest):
    try:
        answer = answer_query_single_call(
            query=request.query,
            language=request.language
        )

        return JSONResponse({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        print(f"Query error: {traceback.format_exc()}")
        return error_response("Failed to answer query", status_code=500)



@app.post("/api/ocr")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from image using OCR"""
    try:
        content = await file.read()
        
        if not is_image_file(content):
            return error_response(
                "Invalid image file. Please upload a valid image (PNG, JPG, JPEG).",
                status_code=400
            )
        
        image = Image.open(io.BytesIO(content))
        text = extract_text_from_image(image)
        
        return JSONResponse({
            "success": True,
            "text": text
        })
    except HTTPException as he:
        return error_response(he.detail, status_code=he.status_code)
    except Exception as e:
        print(f"Unexpected error in extract_text: {traceback.format_exc()}")
        return error_response("Failed to extract text from image.", status_code=500)


@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    """Summarize or explain text"""
    try:
        summary = process_text(request.text, request.explain)
        translation = None
        
        if request.language and request.language.lower() != "none":
            translation = translate_text(summary, request.language)
        
        return JSONResponse({
            "success": True,
            "summary": summary,
            "translation": translation
        })
    except HTTPException as he:
        return error_response(he.detail, status_code=he.status_code)
    except Exception as e:
        print(f"Unexpected error in summarize: {traceback.format_exc()}")
        return error_response("Failed to process text.", status_code=500)


@app.post("/api/translate")
async def translate(request: TranslateRequest):
    """Translate text to specified language"""
    try:
        translation = translate_text(request.text, request.language)
        
        return JSONResponse({
            "success": True,
            "translation": translation,
            "language": request.language
        })
    except HTTPException as he:
        return error_response(he.detail, status_code=he.status_code)
    except Exception as e:
        print(f"Unexpected error in translate: {traceback.format_exc()}")
        return error_response("Translation service unavailable.", status_code=500)


@app.post("/api/visualize")
async def visualize_query(request: QueryRequest):
    """Generate image based on query with specified style"""
    try:
        # FIX: Get style from request (defaults to "realistic" if not provided)
        style = request.style if request.style else "realistic"
        
        # Generate single image prompt
        prompts = generate_image_prompts(request.query, count=1)
        img_prompt = prompts[0] if prompts else request.query
        
        # FIX: Include style in the enhanced prompt
        enhanced_prompt = f"3D ultra HD vibrant, {style}, {img_prompt}"
        print(f"Generating visualization with style '{style}': {enhanced_prompt}")
        
        img_bytes, flag = IG(enhanced_prompt)
        
        if not flag or not img_bytes:
            return error_response(
                "Image generation failed. Please try again with a different prompt.",
                status_code=500
            )
        
        # Convert to base64
        img_base64 = bytes_to_base64(img_bytes)
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "prompt": img_prompt,
            "style": style  # FIX: Return the style used
        })
    except HTTPException as he:
        return error_response(he.detail, status_code=he.status_code)
    except Exception as e:
        print(f"Unexpected error in visualize_query: {traceback.format_exc()}")
        return error_response("Image generation service unavailable.", status_code=500)


@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    language: str = Form("hindi"),
    explain: bool = Form(False),
    generate_images: bool = Form(False),
    style: str = Form("realistic")  # FIX: Added style parameter
):
    """
    Process uploaded file (image or PDF) - extract, summarize, translate, and optionally visualize
    
    Logic:
    - explain=True: Returns explanation (HTML formatted), translation (if language != none)
    - generate_images=True: Returns summary and 4 generated images with specified style
    - Both True: Returns explanation, translation, image prompts, and 4 generated images
    """
    try:
        content = await file.read()
        
        # Step 1: Extract text from file
        if is_image_file(content):
            image = Image.open(io.BytesIO(content))
            extracted_text = extract_text_from_image(image)
        else:
            extracted_text = read_pdf(content)
        
        if not extracted_text:
            return error_response(
                "No text could be extracted from file",
                status_code=400
            )
        
        # Step 2: Process text (summarize or explain)
        summary = process_text(extracted_text, explain)
        
        # Step 3: Translate if requested
        translation = None
        if language and language.lower() != "none":
            translation = translate_text(summary, language)
        
        # Step 4: Generate images if requested
        image_prompts = []
        images = []
        
        if generate_images:
            # Generate prompts based on summary
            image_prompts = generate_image_prompts(summary, count=4)
            
            # FIX: Pass style parameter to image generation
            images = await generate_images_from_prompts(image_prompts, style)
        
        # Build response based on flags
        response_data = {
            "success": True,
            "extracted_text": extracted_text,
            "summary": summary
        }
        
        if translation:
            response_data["translation"] = translation
        
        if generate_images:
            response_data["image_prompts"] = image_prompts
            response_data["images"] = images
            response_data["images_generated"] = len(images)
            response_data["style"] = style  # FIX: Return the style used
        
        return JSONResponse(response_data)
        
    except HTTPException as he:
        return error_response(he.detail, status_code=he.status_code)
    except Exception as e:
        print(f"Unexpected error in process_file: {traceback.format_exc()}")
        return error_response("Failed to process file. Please try again.", status_code=500)


@app.post("/api/explain-image")
async def explain_image(
    file: UploadFile = File(...),
    language: str = Form("hindi")  # "english", "hindi", "none"
):
    try:
        content = await file.read()

        if not is_image_file(content):
            return error_response(
                "Invalid image file. Please upload PNG, JPG, or JPEG.",
                status_code=400
            )

        # Single Vision Call
        html_response = explain_image_single_call(
            image_bytes=content,
            language=language
        )

        return JSONResponse({
            "success": True,
            "explanation": html_response
        })

    except Exception as e:
        print(f"Explain image error: {e}")
        return error_response("Failed to explain image", 500)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
