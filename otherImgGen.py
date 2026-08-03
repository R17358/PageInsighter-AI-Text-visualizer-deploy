"""
Image Generation Module
Handles API calls to external image generation service (Pollinations.ai)
"""

import requests
import os
from urllib.parse import quote
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pollinations.ai base URL (no API key required)
POLLINATIONS_BASE_URL = os.getenv("ImgGenURL")


def ImageGenerator(prompt: str):
    """
    Generate image from text prompt using Pollinations.ai (free, unlimited)
    
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

        # URL-encode the prompt
        encoded_prompt = quote(prompt)

        # Build final URL with generation params
        url = f"{POLLINATIONS_BASE_URL}{encoded_prompt}"
        params = {
            "width": 1024,
            "height": 1024,
            "nologo": "true",
            # "seed": 42,        # uncomment for reproducible output
            # "model": "flux",   # uncomment to force a specific model
        }

        print(f"Requesting image generation for: {prompt[:50]}...")

        # Make API request with timeout
        response = requests.get(
            url,
            params=params,
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
        print("Error: Could not connect to image generation API")
        return None, False

    except requests.exceptions.RequestException as req_error:
        print(f"Error: Request failed - {req_error}")
        return None, False

    except Exception as e:
        print(f"Unexpected error in ImageGenerator: {e}")
        import traceback
        traceback.print_exc()
        return None, False