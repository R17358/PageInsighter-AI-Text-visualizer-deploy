"""
PageInsighter API - Unified Backend
Combines document processing, AI chat, tool calling, and image generation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import google.generativeai as genai
from PIL import Image
import PyPDF2
import io
import base64
from dotenv import load_dotenv
import traceback
import json
from datetime import datetime

# Import custom modules (ensure these exist in your project)
from otherImgGen import ImageGenerator as IG
from image_to_text import ImageToText, explain_image_single_call

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="PageInsighter Unified API",
    version="3.0.0",
    description="AI-powered document processing, chat, and image generation API"
)

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
GEMINI_MODEL = os.getenv("gemini_model", "gemini-2.0-flash-exp")
genai.configure(api_key=API_KEY)

# Session storage (use Redis/database in production)
chat_sessions = {}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "hindi"
    style: Optional[str] = "realistic"

class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = "hindi"
    explain: bool = False

class TranslateRequest(BaseModel):
    text: str
    language: str = "hindi"

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[int] = None

# ============================================================================
# TOOL DEFINITIONS FOR GEMINI
# ============================================================================

AVAILABLE_TOOLS = [
    {
        "name": "generate_image",
        "description": "Generate an image from a text description. Use when user asks to create, generate, or visualize an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate"
                },
                "style": {
                    "type": "string",
                    "description": "Art style for the image",
                    "enum": ["realistic", "anime", "fantasy art", "cyberpunk", "oil painting", "watercolor"]
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Use when user asks about recent events or needs up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "analyze_document",
        "description": "Analyze uploaded documents or images.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_index": {
                    "type": "integer",
                    "description": "Index of the file to analyze (0-based)"
                }
            },
            "required": ["file_index"]
        }
    }
]

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
            detail="API quota exceeded. Please try again later."
        )
    elif "401" in error_msg or "unauthorized" in error_msg:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Check your Gemini API configuration."
        )
    elif "400" in error_msg:
        raise HTTPException(
            status_code=400,
            detail="Invalid request to Gemini API."
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="AI service temporarily unavailable."
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
            detail="Failed to extract text from PDF."
        )

def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Image conversion error: {str(e)}")
        return base64.b64encode(image_bytes).decode('utf-8')

# ============================================================================
# GEMINI API FUNCTIONS
# ============================================================================

def call_gemini_api(prompt: str, image: Optional[Image.Image] = None) -> str:
    """Unified Gemini API call with error handling"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
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

def answer_query_single_call(query: str, language: str | None) -> str:
    """Generate answer with optional translation"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    lang = (language or "english").lower()
    
    if lang in ["none", "no", "no translation", "english"]:
        prompt = f"""
You are an intelligent tutor.

TASK: Answer the user's question accurately and clearly.

CONTENT RULES:
- If mathematical/logical → explain step-by-step and give final answer
- If conceptual → explain clearly with structure
- If short factual → answer concisely

OUTPUT FORMAT (STRICT):
- Output MUST be valid HTML ONLY
- Use inline CSS
- Background: #F5F5F5
- Text color: #000000
- Border-radius: 8px
- Padding: 16px
- Use: <h2>, <p>, <ul>, <li>, <b>

DO NOT add markdown or text outside HTML.

User Question: {query}
"""
    else:
        prompt = f"""
You are an intelligent tutor.

TASK: Answer the user's question accurately and clearly.

LANGUAGE: Translate the FINAL answer into {language}

CONTENT RULES:
- If mathematical/logical → explain step-by-step and give final answer
- If conceptual → explain clearly with structure

OUTPUT FORMAT (STRICT):
- Output MUST be valid HTML ONLY
- DO NOT include <html>, <head>, <body>, or <style> tags
- Use INLINE CSS only on elements (style="...")
- All styles MUST be container-safe (no global selectors)

DESIGN RULES:
- Background color: light neutral tone (e.g. #E6F2F2, #ECEFF1, #F0F4F4)
- Text color: dark colors ONLY (choose from: #0F172A, #0B3C3D, #1F2933, #2C2C2C, #3A2E2E)
- Headings color: dark cyan or deep teal (e.g. #0B6B6B, #065F5B)
- NEVER use pure white (#FFFFFF) or pure black (#000000)
- Border-radius: 8px
- Padding: 16px
- Font-family: inherit

ALLOWED TAGS ONLY:
- <div>, <h2>, <p>, <ul>, <li>, <b>

CONTENT RULES:
- No background or color styles on body or root
- No external fonts, scripts, or images
- No animations or transitions


DO NOT add markdown.

User Question: {query}
"""
    
    response = model.generate_content(prompt)
    
    if not response or not response.text:
        raise Exception("Empty response from Gemini")
    
    return response.text.strip()

def process_text(text: str, explain: bool = False) -> str:
    """Summarize or explain text"""
    if explain:
        prompt = f"""Determine if the input is a mathematical problem or descriptive text.
- If mathematical → explain steps, provide calculations, and final answer
- If descriptive → provide comprehensive explanation

Write everything in HTML format with inline styles.
Use proper formatting for clarity and readability.

Styling:
OUTPUT FORMAT (STRICT):
- Output MUST be valid HTML ONLY
- DO NOT include <html>, <head>, <body>, or <style> tags
- Use INLINE CSS only on elements (style="...")
- All styles MUST be container-safe (no global selectors)

DESIGN RULES:
- Background color: light neutral tone (e.g. #E6F2F2, #ECEFF1, #F0F4F4)
- Text color: dark colors ONLY (choose from: #0F172A, #0B3C3D, #1F2933, #2C2C2C, #3A2E2E)
- Headings color: dark cyan or deep teal (e.g. #0B6B6B, #065F5B)
- NEVER use pure white (#FFFFFF) or pure black (#000000)
- Border-radius: 8px
- Padding: 16px
- Font-family: inherit

ALLOWED TAGS ONLY:
- <div>, <h2>, <p>, <ul>, <li>, <b>

CONTENT RULES:
- No background or color styles on body or root
- No external fonts, scripts, or images
- No animations or transitions


Input: {text}"""
    else:
        prompt = f"Summarize the following text concisely:\n\n{text}"
    
    return call_gemini_api(prompt)

def translate_text(text: str, language: str) -> str:
    """Translate text to specified language"""
    prompt = f"Translate the following text into {language}:\n\n{text}"
    return call_gemini_api(prompt)

def generate_image_prompts(text: str, count: int = 4) -> List[str]:
    """Generate multiple image generation prompts"""
    prompt = f"""Generate {count} diverse, detailed image generation prompts based on this text.

Each prompt should:
- Be visually descriptive and unique
- Capture different aspects from the content
- Be optimized for 3D ultra HD vibrant image generation
- Be 15-25 words long

Return ONLY a JSON array of strings, no additional text.

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
        print(f"JSON parsing error: {e}, using fallback")
        single_prompt = f"A detailed, vibrant 3D visualization: {text[:200]}"
        return [single_prompt]

# ============================================================================
# TOOL EXECUTION FUNCTIONS
# ============================================================================

async def execute_generate_image(prompt: str, style: str = "realistic") -> dict:
    """Execute image generation tool"""
    try:
        enhanced_prompt = f"3D ultra HD vibrant, {style}, {prompt}"
        
        img_bytes, flag = IG(enhanced_prompt)
        
        if flag and img_bytes:
            img_base64 = bytes_to_base64(img_bytes)
            return {
                "success": True,
                "image": img_base64,
                "prompt": enhanced_prompt
            }
        else:
            return {
                "success": False,
                "error": "Image generation failed"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def execute_web_search(query: str) -> dict:
    """Execute web search tool (placeholder)"""
    return {
        "success": True,
        "message": f"Web search simulated for: {query}",
        "query": query,
        "note": "Implement actual search API here"
    }

async def execute_analyze_document(file_content: bytes, file_type: str) -> dict:
    """Analyze uploaded document"""
    try:
        if is_image_file(file_content):
            model = genai.GenerativeModel(GEMINI_MODEL)
            image = Image.open(io.BytesIO(file_content))
            
            prompt = "Extract and describe all text and content from this image."
            response = model.generate_content([prompt, image])
            
            return {
                "success": True,
                "analysis": response.text,
                "type": "image"
            }
        else:
            text = read_pdf(file_content)
            return {
                "success": True,
                "analysis": f"Extracted {len(text)} characters from PDF",
                "text": text[:1000] + "..." if len(text) > 1000 else text,
                "type": "pdf"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def generate_images_from_prompts(prompts: List[str], style: str = "realistic") -> List[dict]:
    """Generate images from prompts with specified style"""
    images = []
    print(f"\nGenerating {len(prompts)} images with style: {style}...")
    
    for i, base_prompt in enumerate(prompts):
        try:
            enhanced_prompt = f"3D ultra HD vibrant, {style}, {base_prompt}"
            print(f"  [{i+1}/{len(prompts)}] Generating: {enhanced_prompt[:60]}...")
            
            img_bytes, flag = IG(enhanced_prompt)
            
            if flag and img_bytes:
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
    
    print(f"Successfully generated {len(images)}/{len(prompts)} images\n")
    return images

# ============================================================================
# CHAT ENDPOINT WITH TOOL CALLING
# ============================================================================

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    session_id: str = Form(...),
    history: str = Form("[]"),
    files: List[UploadFile] = File(default=[])
):
    """Main chat endpoint with memory and tool calling"""
    try:
        chat_history = json.loads(history)
        
        # Initialize session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "uploaded_files": []
            }
        
        session = chat_sessions[session_id]
        
        # Process uploaded files
        file_contexts = []
        if files:
            for idx, file in enumerate(files):
                content = await file.read()
                analysis = await execute_analyze_document(content, file.content_type)
                
                file_contexts.append({
                    "filename": file.filename,
                    "analysis": analysis
                })
                
                session["uploaded_files"].append({
                    "filename": file.filename,
                    "uploaded_at": datetime.now().isoformat()
                })
        
        # Build system instruction
        system_instruction = """You are PageInsighter AI, an advanced assistant with tools.

Your capabilities:
1. Generate images using generate_image tool
2. Search the web using web_search tool
3. Analyze documents using analyze_document tool

Be helpful, creative, and proactive. Explain what you're doing when using tools.
"""
        
        # Prepare file context
        file_context_text = ""
        if file_contexts:
            file_context_text = "\n\nUploaded Files:\n"
            for fc in file_contexts:
                file_context_text += f"- {fc['filename']}: {fc['analysis'].get('analysis', 'No analysis')}\n"
        
        # Build conversation history
        gemini_history = [{
            "role": "model",
            "parts": [system_instruction]
        }]
        
        for msg in chat_history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Create model and chat
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        chat = model.start_chat(history=gemini_history)
        
        # Send user message
        user_message = message
        if file_context_text:
            user_message += file_context_text
        
        response = chat.send_message(user_message)
        
        # Check for tool calls (simplified)
        tool_calls = []
        final_response = response.text
        
        if "generate_image" in message.lower() or "create image" in message.lower():
            tool_result = await execute_generate_image(message, "realistic")
            tool_calls.append({
                "name": "generate_image",
                "result": tool_result
            })
        
        if "search" in message.lower() and "web" in message.lower():
            tool_result = await execute_web_search(message)
            tool_calls.append({
                "name": "web_search",
                "result": tool_result
            })
        
        # Save to session
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        session["messages"].append({
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.now().isoformat(),
            "tool_calls": tool_calls if tool_calls else None
        })
        
        return JSONResponse({
            "success": True,
            "response": final_response,
            "tool_calls": tool_calls,
            "session_id": session_id,
            "message_count": len(session["messages"])
        })
        
    except Exception as e:
        print(f"Chat error: {traceback.format_exc()}")
        return error_response(f"Chat failed: {str(e)}", 500)

# ============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# ============================================================================

@app.post("/api/query")
async def answer_query(request: QueryRequest):
    """Simple query endpoint"""
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
        return error_response("Failed to answer query", 500)

@app.post("/api/ocr")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from image using OCR"""
    try:
        content = await file.read()
        
        if not is_image_file(content):
            return error_response("Invalid image file", 400)
        
        image = Image.open(io.BytesIO(content))
        text = extract_text_from_image(image)
        
        return JSONResponse({
            "success": True,
            "text": text
        })
    except HTTPException as he:
        return error_response(he.detail, he.status_code)
    except Exception as e:
        print(f"OCR error: {traceback.format_exc()}")
        return error_response("Failed to extract text", 500)

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
    except Exception as e:
        print(f"Summarize error: {traceback.format_exc()}")
        return error_response("Failed to process text", 500)

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
        print(f"Translate error: {traceback.format_exc()}")
        return error_response("Translation failed", 500)

@app.post("/api/visualize")
async def visualize_query(request: QueryRequest):
    """Generate image based on query"""
    try:
        style = request.style if request.style else "realistic"
        
        prompts = generate_image_prompts(request.query, count=1)
        img_prompt = prompts[0] if prompts else request.query
        
        enhanced_prompt = f"3D ultra HD vibrant, {style}, {img_prompt}"
        print(f"Generating visualization: {enhanced_prompt}")
        
        img_bytes, flag = IG(enhanced_prompt)
        
        if not flag or not img_bytes:
            return error_response("Image generation failed", 500)
        
        img_base64 = bytes_to_base64(img_bytes)
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "prompt": img_prompt,
            "style": style
        })
    except Exception as e:
        print(f"Visualize error: {traceback.format_exc()}")
        return error_response("Image generation failed", 500)

@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    language: str = Form("hindi"),
    explain: bool = Form(False),
    generate_images: bool = Form(False),
    style: str = Form("realistic")
):
    """Process uploaded file - extract, summarize, translate, and optionally visualize"""
    try:
        content = await file.read()
        
        # Extract text
        if is_image_file(content):
            image = Image.open(io.BytesIO(content))
            extracted_text = extract_text_from_image(image)
        else:
            extracted_text = read_pdf(content)
        
        if not extracted_text:
            return error_response("No text extracted from file", 400)
        
        # Process text
        summary = process_text(extracted_text, explain)
        
        # Translate if requested
        translation = None
        if language and language.lower() != "none":
            translation = translate_text(summary, language)
        
        # Generate images if requested
        image_prompts = []
        images = []
        
        if generate_images:
            image_prompts = generate_image_prompts(summary, count=4)
            images = await generate_images_from_prompts(image_prompts, style)
        
        # Build response
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
            response_data["style"] = style
        
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Process file error: {traceback.format_exc()}")
        return error_response("Failed to process file", 500)

@app.post("/api/explain-image")
async def explain_image(
    file: UploadFile = File(...),
    language: str = Form("hindi")
):
    """Explain image content"""
    try:
        content = await file.read()
        
        if not is_image_file(content):
            return error_response("Invalid image file", 400)
        
        html_response = explain_image_single_call(
            image_bytes=content,
            language=language
        )
        
        return JSONResponse({
            "success": True,
            "explanation": html_response
        })
    except Exception as e:
        print(f"Explain image error: {traceback.format_exc()}")
        return error_response("Failed to explain image", 500)

@app.post("/api/generate-image-prompt")
async def generate_image_prompt(request: QueryRequest):
    """Generate optimized image prompt"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""Generate a detailed image generation prompt based on this text:
{request.query}

Make it descriptive, visual, and optimized for AI image generation.
Keep it under 50 words."""
        
        response = model.generate_content(prompt)
        
        return JSONResponse({
            "success": True,
            "prompt": response.text.strip()
        })
    except Exception as e:
        return error_response("Prompt generation failed", 500)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@app.delete("/api/chat/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return JSONResponse({"success": True, "message": "Session cleared"})
    return JSONResponse({"success": False, "message": "Session not found"}, status_code=404)

@app.get("/api/chat/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session details"""
    if session_id in chat_sessions:
        return JSONResponse({
            "success": True,
            "session": chat_sessions[session_id]
        })
    return JSONResponse({"success": False, "message": "Session not found"}, status_code=404)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PageInsighter Unified API is running",
        "version": "3.0.0",
        "status": "healthy",
        "features": [
            "AI Chat with Memory",
            "Tool Calling (Image Gen, Web Search)",
            "File Analysis (PDF, Images)",
            "Document Processing",
            "Translation",
            "OCR",
            "Image Generation"
        ],
        "model": GEMINI_MODEL,
        "endpoints": {
            "chat": "/api/chat",
            "query": "/api/query",
            "ocr": "/api/ocr",
            "summarize": "/api/summarize",
            "translate": "/api/translate",
            "visualize": "/api/visualize",
            "process_file": "/api/process-file",
            "explain_image": "/api/explain-image",
            "generate_image_prompt": "/api/generate-image-prompt"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_configured": bool(API_KEY),
        "model": GEMINI_MODEL,
        "active_sessions": len(chat_sessions)
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 PageInsighter Unified API Starting...")
    print("="*70)
    print(f"📍 Server: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🤖 Model: {GEMINI_MODEL}")
    print(f"✨ Features: Chat, OCR, Translation, Image Gen, Document Processing")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)