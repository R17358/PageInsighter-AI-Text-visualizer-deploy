# ans = ImageToText("assets/village.jpg")
# print(ans)
import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_VISION_MODEL")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = genai.GenerativeModel(MODEL_NAME)

def ImageToText(image_path):
    """
    Takes an image path and returns text description.
    If image contains a math problem, it solves it.
    """
    img = Image.open(image_path)

    prompt = """
    Describe the image clearly.
    If the image contains a mathematical problem, solve it step by step.
    """

    response = model.generate_content([prompt, img])

    return response.text.strip()


# Example usage
# ans = ImageToText("assets/village.jpg")
# print(ans)
