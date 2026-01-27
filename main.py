from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import os
import numpy as np
import google.generativeai as genai
from PIL import Image
import PyPDF2
import pytesseract
import io
import base64
from dotenv import load_dotenv

# Import your custom modules
from otherImgGen import ImageGenerator as IG
from image_to_text import ImageToText

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io

load_dotenv()

app = FastAPI(title="PageInsighter API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_KEY = os.getenv("gemini_key")
gemini_model = os.getenv("gemini_model")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(gemini_model)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "hindi"

class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = "hindi"
    explain: bool = False

class TranslateRequest(BaseModel):
    text: str
    language: str = "hindi"

class ImagePromptRequest(BaseModel):
    text: str

# Helper functions
def recognize_text(image_array: np.ndarray) -> str:
    """Extract text from image using OCR"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='hin+eng')
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Error: {str(e)}")

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
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Error: {str(e)}")

def generate_answer(query: str) -> str:
    """Generate answer using Gemini"""
    try:
        prompt = f"""Please provide a detailed response to the following query in HTML format: {query}. 
The response should use proper inline CSS styles for formatting and be well-structured. 

Ensure the HTML is visually appealing and easy to read. 

Additionally, implement the following dynamic styling for readability:
- set the font color to dark (e.g., `#000000`),
- set the background color light (e.g., `#FFFFFF`, `#F0F0F0`), 
- set the background border-radius of 8px.

This ensures that the text is always readable regardless of the background color."""
        
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

def summarize_text(text: str, explain: bool = False) -> str:
    """Summarize or explain text"""
    try:
        if explain:
            prompt = f"""Determine if the input is a mathematical problem or descriptive text. 
- If it is a mathematical problem, explain the steps clearly, provide detailed calculations, and include the final answer. 
- If it is descriptive text, provide summary of {text}.

Write everything in HTML format with inline styles for rendering. 
Ensure proper formatting for clarity and readability, using appropriate headings, paragraphs, and lists where necessary. 
The response should be visually appealing, with attention to font size, color, and layout to ensure ease of understanding based on the type of input. 

Additionally, implement the following dynamic styling for readability:
- set the font color to dark (e.g., `#000000`).
- set the background color light (e.g., `#FFFFFF`, `#F0F0F0`)
- set the background border-radius of 8px.

The input to be analyzed is: {text}."""
        else:
            prompt = f"Summarize the following text: {text}"
        
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization Error: {str(e)}")

def generate_image_prompt(text: str) -> str:
    """Generate prompt for image generation"""
    try:
        response = model.generate_content(
            f"Generate a detailed prompt for image generation for the given text: {text}. Only generate the single best prompt and nothing else."
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt Generation Error: {str(e)}")

def translate_text(text: str, language: str = 'hindi') -> str:
    """Translate text to specified language"""
    try:
        response = model.generate_content(
            f"Translate the following {text} into {language} and use simple words."
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation Error: {str(e)}")

# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PageInsighter API is running", "version": "1.0.0"}

@app.post("/api/query")
async def answer_query(request: QueryRequest):
    """Answer a general query using Gemini"""
    try:
        answer = generate_answer(request.query)
        translation = None
        
        if request.language and request.language.lower() != "none":
            translation = translate_text(answer, request.language)
        
        return JSONResponse({
            "success": True,
            "answer": answer,
            "translation": translation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-image-prompt")
async def create_image_prompt(request: ImagePromptRequest):
    """Generate image generation prompt from text"""
    try:
        prompt = generate_image_prompt(request.text)
        return JSONResponse({
            "success": True,
            "prompt": prompt
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize")
async def visualize_query(request: QueryRequest):
    """Generate image based on query"""
    try:
        img_prompt = generate_image_prompt(request.query)
        enhanced_prompt = f"3D ultra HD vibrant {img_prompt}"
        
        img, flag = IG(enhanced_prompt)
        
        # Convert PIL image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "prompt": img_prompt
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from image using OCR"""
    try:
        content = await file.read()
        
        if not is_image_file(content):
            raise HTTPException(status_code=400, detail="File is not a valid image")
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(content))
        image_array = np.array(image)
        
        text = recognize_text(image_array)
        
        return JSONResponse({
            "success": True,
            "text": text
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    """Summarize or explain text"""
    try:
        summary = summarize_text(request.text, request.explain)
        translation = None
        
        if request.language and request.language.lower() != "none":
            translation = translate_text(summary, request.language)
        
        return JSONResponse({
            "success": True,
            "summary": summary,
            "translation": translation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    language: str = Form("hindi"),
    explain: bool = Form(False),
    generate_images: bool = Form(True)
):
    """Process uploaded file (image or PDF) - extract, summarize, translate, and optionally visualize"""
    try:
        content = await file.read()
        
        # Determine file type and extract text
        if is_image_file(content):
            image = Image.open(io.BytesIO(content))
            image_array = np.array(image)
            extracted_text = recognize_text(image_array)
        else:
            extracted_text = read_pdf(content)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        # Summarize
        summary = summarize_text(extracted_text, explain)
        
        # Translate
        translation = None
        if language and language.lower() != "none":
            translation = translate_text(summary, language)
        
        # Generate images
        images = []
        if generate_images and not explain:
            for i in range(4):
                img_prompt = generate_image_prompt(summary)
                enhanced_prompt = f"3D ultra HD vibrant {img_prompt}"
                img, flag = IG(enhanced_prompt)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                images.append({
                    "prompt": img_prompt,
                    "image": img_base64
                })
        
        return JSONResponse({
            "success": True,
            "extracted_text": extracted_text,
            "summary": summary,
            "translation": translation,
            "images": images
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain-image")
async def explain_image(
    file: UploadFile = File(...),
    language: str = Form("hindi")
):
    try:
        content = await file.read()

        if not is_image_file(content):
            raise HTTPException(status_code=400, detail="File is not a valid image")

        print("before img to text")
        # Step 1: Image → Text (Vision)
        explanation = ImageToText(content)

        print("\n After ing to text")

        print(f"\n {explanation}")

        # Step 2: Text → Detailed HTML explanation
        detailed_prompt = f"""
You are an intelligent tutor.

Determine whether the input below is:
1) A mathematical problem → explain step-by-step with calculations and final answer
2) Descriptive/visual content → explain clearly and structurally

Rules:
- Output MUST be valid HTML
- Use inline CSS
- Background: #F5F5F5
- Text color: #000000
- Border-radius: 8px
- Padding: 16px
- Use <h2>, <p>, <ul>, <li>, <b>

Input:
{explanation}
"""

        detailed_explanation = model.generate_content(detailed_prompt).text

        # Step 3: Optional Translation
        translation = None
        if language and language.lower() != "none":
            translation = translate_text(detailed_explanation, language)

        return JSONResponse({
            "success": True,
            "explanation": detailed_explanation,
            "translation": translation
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
