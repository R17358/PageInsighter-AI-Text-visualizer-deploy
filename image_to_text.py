"""
Image to Text Module
Converts images to descriptive text using Gemini Vision API
"""

import google.generativeai as genai
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("gemini_key")
GEMINI_MODEL = os.getenv("gemini_model", "gemini-pro-vision")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: gemini_key not found in environment variables")


def ImageToText(image_data):
    """
    Convert image to descriptive text using Gemini Vision
    
    Args:
        image_data: Either bytes or PIL Image object
        
    Returns:
        str: Detailed description of the image content
        
    Raises:
        Exception: If API call fails or image processing fails
    """
    try:
        # Convert bytes to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create vision model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Detailed prompt for comprehensive analysis
        prompt = """
Analyze this image in detail and provide a comprehensive description.

Include:
1. Main subject/focus of the image
2. Any text visible in the image (extract it exactly)
3. Visual elements (colors, shapes, objects)
4. Context or setting
5. Any mathematical equations, diagrams, or technical content
6. Overall composition and notable details

If this is:
- A document/page: Extract all text verbatim
- A mathematical problem: Describe the problem clearly
- A diagram/chart: Explain what it represents
- A photo: Describe the scene and subjects

Be thorough and precise in your description.
"""
        
        # Generate content with image
        response = model.generate_content([prompt, image])
        
        if not response or not response.text:
            raise ValueError("Empty response from Gemini Vision API")
        
        description = response.text.strip()
        
        # Validate response
        if len(description) < 10:
            raise ValueError("Description too short, likely an error")
        
        return description
        
    except Exception as e:
        error_msg = f"Failed to analyze image: {str(e)}"
        print(f"Error in ImageToText: {error_msg}")
        raise Exception(error_msg)


# Test function
if __name__ == "__main__":
    print("ImageToText module loaded successfully")
    print("To test, call ImageToText(image_bytes) or ImageToText(pil_image)")