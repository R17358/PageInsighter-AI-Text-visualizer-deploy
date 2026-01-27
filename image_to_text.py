# ans = ImageToText("assets/village.jpg")
# print(ans)
import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import io

# Load env variables
load_dotenv()

GEMINI_API_KEY = os.getenv("gemini_key")
MODEL_NAME = os.getenv("GEMINI_VISION_MODEL")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = genai.GenerativeModel(MODEL_NAME)

def ImageToText(image_bytes):
    """
    Accepts image bytes or BytesIO and returns extracted understanding
    """
    if isinstance(image_bytes, bytes):
        img = Image.open(io.BytesIO(image_bytes))
    else:
        img = Image.open(image_bytes)

    prompt = """
    Describe the image clearly.
    If the image contains a mathematical problem, solve it step by step.
    """
    print("Response ...")
    response = model.generate_content([prompt, img])
    print(response)
    return response.text.strip()


# Example usage
# ans = ImageToText("assets/village.jpg")
# print(ans)
