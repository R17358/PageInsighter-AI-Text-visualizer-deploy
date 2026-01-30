
"""
PageInsighter API - Document Processing & AI Chat with Tools
Enhanced with conversational AI chat using Gemini 2.5 Flash and tool calling
"""

# uvicorn main:app 

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

# Import custom modules (these should exist in your project)
# from otherImgGen import ImageGenerator as IG
# from image_to_text import ImageToText, explain_image_single_call

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PageInsighter AI Chat API", 
    version="3.0.0",
    description="Advanced AI chat with memory, file analysis, and tool calling"
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
GEMINI_MODEL = os.getenv("gemini_model")  # Using Gemini 2.5 Flash
genai.configure(api_key=API_KEY)

# Session storage (in production, use Redis or database)
chat_sessions = {}

# ============================================================================
# TOOL DEFINITIONS FOR GEMINI
# ============================================================================

# Define tools for Gemini to use
tools = [
    {
        "name": "generate_image",
        "description": "Generate an image from a text description. Use this when user asks to create, generate, or visualize an image.",
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
        "description": "Search the web for current information. Use when user asks about recent events, news, or needs up-to-date information.",
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
        "description": "Analyze uploaded documents or images. This is automatically called when files are uploaded.",
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
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "hindi"
    style: Optional[str] = "realistic"

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[int] = None

# ============================================================================
# HELPER FUNCTIONS
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
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

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
# TOOL EXECUTION FUNCTIONS
# ============================================================================

async def execute_generate_image(prompt: str, style: str = "realistic") -> dict:
    """Execute image generation tool"""
    try:
        # Here you would call your actual image generation function
        # For demo, we'll return a placeholder
        enhanced_prompt = f"3D ultra HD vibrant, {style}, {prompt}"
        
        # Uncomment when you have the actual function
        # from otherImgGen import ImageGenerator as IG
        # img_bytes, flag = IG(enhanced_prompt)
        # if flag and img_bytes:
        #     img_base64 = bytes_to_base64(img_bytes)
        #     return {"success": True, "image": img_base64, "prompt": enhanced_prompt}
        
        # Placeholder response
        return {
            "success": True,
            "message": f"Image generation simulated for: {enhanced_prompt}",
            "prompt": enhanced_prompt
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def execute_web_search(query: str) -> dict:
    """Execute web search tool"""
    try:
        # Here you would call actual web search API
        # For demo, returning placeholder
        return {
            "success": True,
            "message": f"Web search simulated for: {query}",
            "query": query,
            "note": "In production, this would return actual search results"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def execute_analyze_document(file_content: bytes, file_type: str) -> dict:
    """Analyze uploaded document"""
    try:
        if is_image_file(file_content):
            # Extract text from image using Gemini Vision
            model = genai.GenerativeModel(GEMINI_MODEL)
            image = Image.open(io.BytesIO(file_content))
            
            prompt = "Extract and describe all text and content from this image in detail."
            response = model.generate_content([prompt, image])
            
            return {
                "success": True,
                "analysis": response.text,
                "type": "image"
            }
        else:
            # PDF processing
            text = read_pdf(file_content)
            return {
                "success": True,
                "analysis": f"Extracted {len(text)} characters from PDF",
                "text": text[:1000] + "..." if len(text) > 1000 else text,
                "type": "pdf"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

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
    """
    Main chat endpoint with memory and tool calling
    """
    try:
        # Parse history
        chat_history = json.loads(history)
        
        # Initialize session if new
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
        
        # Build context for Gemini
        system_instruction = """You are PageInsighter AI, an advanced AI assistant with access to powerful tools.

Your capabilities:
1. Generate images from text descriptions using the generate_image tool
2. Search the web for current information using the web_search tool  
3. Analyze uploaded documents and images using the analyze_document tool

When users ask you to:
- Create/generate/visualize images → use generate_image tool
- Find current info/news/search → use web_search tool
- Analyze their uploaded files → use analyze_document tool

Be helpful, creative, and proactive in using these tools to assist users.
Always explain what you're doing when you use a tool.
"""
        
        # Prepare file context if files were uploaded
        file_context_text = ""
        if file_contexts:
            file_context_text = "\n\nUploaded Files:\n"
            for fc in file_contexts:
                file_context_text += f"- {fc['filename']}: {fc['analysis'].get('analysis', 'No analysis')}\n"
        
        # Build conversation history for Gemini
        gemini_history = gemini_history = [{
            "role": "model",
            "parts": [system_instruction]
        }]

        for msg in chat_history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        
        # Create model with tools
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            # system_instruction=system_instruction
        )
        
        # Start chat with history
        chat = model.start_chat(history=gemini_history)
        
        # Send user message with file context
        user_message = message
        if file_context_text:
            user_message += file_context_text
        
        # Get response (may include tool calls)
        response = chat.send_message(user_message)
        
        # Check if model wants to use tools
        tool_calls = []
        final_response = response.text
        
        # Note: Actual tool calling implementation would check response.candidates[0].content.parts
        # and execute tools accordingly. This is a simplified version.
        
        # For demo, we'll check if the response mentions tools
        if "generate_image" in message.lower() or "create image" in message.lower():
            # Simulate tool call
            tool_result = await execute_generate_image(message, "realistic")
            tool_calls.append({
                "name": "generate_image",
                "result": f"Generated image based on: {message}"
            })
        
        if "search" in message.lower() and "web" in message.lower():
            tool_result = await execute_web_search(message)
            tool_calls.append({
                "name": "web_search",
                "result": f"Searched web for: {message}"
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
# EXISTING ENDPOINTS (Query, OCR, etc.)
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    print(API_KEY)
    return {
        "message": "PageInsighter AI Chat API is running",
        "version": "3.0.0",
        "status": "healthy",
        "features": [
            "AI Chat with Memory",
            "Tool Calling (Image Gen, Web Search)",
            "File Analysis",
            "Document Processing",
            "Translation"
        ],
        "model": GEMINI_MODEL,
        "endpoints": {
            "chat": "/api/chat",
            "query": "/api/query",
            "visualize": "/api/visualize"
        }
    }

@app.post("/api/query")
async def answer_query(request: QueryRequest):
    """Simple query endpoint"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""Answer this question clearly and concisely: {request.query}

Format your response in HTML with inline styles:
- Background: #F5F5F5
- Text color: #000000
- Use proper headings and structure
"""
        
        if request.language and request.language.lower() != "none":
            prompt += f"\n\nTranslate the answer to {request.language}."
        
        response = model.generate_content(prompt)
        
        return JSONResponse({
            "success": True,
            "answer": response.text
        })
    except Exception as e:
        print(f"Query error: {traceback.format_exc()}")
        return error_response("Failed to answer query", 500)

@app.post("/api/visualize")
async def visualize_query(request: QueryRequest):
    """Generate image from text"""
    try:
        # Placeholder - replace with actual image generation
        result = await execute_generate_image(request.query, request.style)
        
        return JSONResponse({
            "success": True,
            "message": result.get("message"),
            "prompt": result.get("prompt"),
            "style": request.style,
            # "image": result.get("image")  # Uncomment when using real image gen
        })
    except Exception as e:
        print(f"Visualize error: {traceback.format_exc()}")
        return error_response("Image generation failed", 500)

@app.post("/api/generate-image-prompt")
async def generate_image_prompt(request: QueryRequest):
    """Generate optimized image prompt"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""Generate a detailed, vivid image generation prompt based on this text:
{request.query}

Make it descriptive, visual, and optimized for AI image generation.
Keep it under 50 words.
"""
        
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 PageInsighter AI Chat API Starting...")
    print("="*70)
    print(f"📍 Server: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🤖 Model: {GEMINI_MODEL}")
    print(f"🔧 Tools: Image Generation, Web Search, Document Analysis")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
