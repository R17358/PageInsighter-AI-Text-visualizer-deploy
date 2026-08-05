"""
Image Generation Module
Handles API calls to Hugging Face Inference Providers (FLUX.1 Schnell)
"""

import os
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_TOKEN")

# Initialize Hugging Face Client
client = InferenceClient(
    provider="auto",   # Automatically selects an available provider
    api_key=HF_API_KEY
)

MODEL = "black-forest-labs/FLUX.1-schnell"


def ImageGenerator(prompt: str):
    """
    Generate image from text prompt using Hugging Face Inference Providers

    Args:
        prompt (str): Text description for image generation

    Returns:
        tuple: (image_bytes, success_flag)
            - image_bytes: Raw image bytes
            - success_flag: True if successful
    """

    try:

        if not prompt or not prompt.strip():
            print("Error: Empty prompt provided")
            return None, False

        if not HF_API_KEY:
            print("Error: HF_TOKEN missing in .env")
            return None, False

        print(f"Generating image for: {prompt[:60]}...")

        # Generate PIL Image
        image = client.text_to_image(
            prompt,
            model=MODEL
        )

        # Convert PIL Image → Bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")

        print("✓ Image generated successfully")

        return img_bytes.getvalue(), True

    except Exception as e:

        print(f"Image Generation Error: {e}")

        return None, False