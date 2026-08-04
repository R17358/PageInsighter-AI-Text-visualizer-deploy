"""
Image Generation Module
Handles API calls to external image generation service (Fireworks AI - FLUX.1 Schnell)
"""

import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fireworks AI Configuration
FIREWORKS_BASE_URL = os.getenv("ImgGenURL")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")


def ImageGenerator(prompt: str):
    """
    Generate image from text prompt using Fireworks AI (FLUX.1 Schnell Workflow)
    
    Args:
        prompt (str): Text description for image generation
        
    Returns:
        tuple: (image_bytes, success_flag)
            - image_bytes: Raw image data as bytes (or None if failed)
            - success_flag: Boolean indicating success/failure
    """
    try:
        if not prompt or not prompt.strip():
            print("Error: Empty prompt provided")
            return None, False

        if not FIREWORKS_API_KEY:
            print("Error: FIREWORKS_API_KEY is missing in environment variables")
            return None, False

        if not FIREWORKS_BASE_URL:
            print("Error: ImgGenURL (Fireworks API URL) is missing in environment variables")
            return None, False

        # Fireworks AI expects headers with Bearer token
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json"
        }

        # Request payload for FLUX.1 Schnell
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "num_steps": 4  # Schnell performs best and fastest at 4 steps
        }

        print(f"Requesting image generation from Fireworks AI for: {prompt[:50]}...")

        # Fireworks AI requires a POST request with JSON payload (not a GET request with URL params)
        response = requests.post(
            FIREWORKS_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )

        # Check response status
        if response.status_code == 200:
            image_bytes = response.content

            # Validate image data
            if not image_bytes or len(image_bytes) < 100:
                print("Error: Received empty or invalid image data")
                return None, False

            print(f"✓ Image generated successfully ({len(image_bytes)} bytes)")
            return image_bytes, True

        else:
            print(f"Error: API returned status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None, False

    except requests.exceptions.Timeout:
        print("Error: Image generation request timed out (60s)")
        return None, False

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Fireworks AI API")
        return None, False

    except requests.exceptions.RequestException as req_error:
        print(f"Error: Request failed - {req_error}")
        return None, False

    except Exception as e:
        print(f"Unexpected error in ImageGenerator: {e}")
        import traceback
        traceback.print_exc()
        return None, False
