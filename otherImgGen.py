"""
Image Generation Module
Handles API calls to external image generation service
"""

import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv("otherImg_url")
API_KEY = os.getenv("otherImg_key")

# SAVE_DIR = "gen_images"


def ImageGenerator(prompt: str):
    """
    Generate image from text prompt using external API
    
    Args:
        prompt (str): Text description for image generation
        
    Returns:
        tuple: (image_bytes, success_flag)
            - image_bytes: Raw image data as bytes (or None if failed)
            - success_flag: Boolean indicating success/failure
    """
    try:
        # Validate environment variables
        if not API_URL or not API_KEY:
            print("Error: Missing API_URL or API_KEY in environment variables")
            return None, False
        
        # Ensure directory exists for local saving (optional)
        # os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Accept": "image/jpeg",
            "Authorization": f"Bearer {API_KEY}",
        }

        data = {
            "prompt": prompt
        }

        print(f"Requesting image generation for: {prompt[:50]}...")
        
        # Make API request with timeout
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=data,
            timeout=60  # 60 second timeout
        )

        # Check response status
        if response.status_code == 200:
            image_bytes = response.content
            
            # Validate image data
            if not image_bytes or len(image_bytes) < 100:
                print("Error: Received empty or invalid image data")
                return None, False
            
            # Optional: Save to disk for debugging/backup
            # try:
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     # save_path = os.path.join(SAVE_DIR, f"generated_{timestamp}.jpg")
            #     # with open(save_path, "wb") as f:
            #         # f.write(image_bytes)
            #     # print(f"✓ Image saved locally at: {save_path}")
            # except Exception as save_error:
            #     print(f"Warning: Could not save image locally: {save_error}")
            #     # Continue even if saving fails
            
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


# Test function
# if __name__ == "__main__":
#     # Test the image generator
#     test_prompt = "A beautiful sunset over mountains, 3D ultra HD vibrant"
#     img_bytes, success = ImageGenerator(test_prompt)
    
#     if success:
#         print(f"\n✓ Test successful! Generated {len(img_bytes)} bytes")
#     else:
#         print("\n✗ Test failed!")

# # ================== RUN ==================
# img_bytes, status = ImageGenerator("Fantasy image of a cat in the forest")

# if status:
#     # Convert bytes → numpy image
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     print(img)


# Example usage:
# API = "YOUR_API_KEY"
# url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
# imgGen(API, url, "A futuristic city with robots", "./generated_images/city.jpg")


# def query(payload, timeout=30):
#     try:
#         response = requests.post(
#             API_URL, 
#             headers=headers, 
#             json=payload,
#             timeout=timeout
#         )
        
#         # Check status
#         if response.status_code == 503:
#             return {"error": "Model is loading, please retry in 20s"}
#         elif response.status_code == 429:
#             return {"error": "Rate limit exceeded"}
#         elif response.status_code != 200:
#             return {"error": f"API error: {response.status_code}"}
            
#         return response.content
        
#     except requests.exceptions.Timeout:
#         return {"error": "Request timeout"}
#     except requests.exceptions.RequestException as e:
#         return {"error": f"Request failed: {str(e)}"}

# def ImageGenerator(prompt):
#     try:
#         output = query({"inputs": prompt})
        
#         # Check if response is valid
#         if isinstance(output, dict) and "error" in output:
#             return None, False
            
#         image = Image.open(io.BytesIO(output))
#         # Optional: save locally if needed
#         # filename = f'image_{int(time.time())}.jpg'
#         # image.save(filename)
        
#         return image, True  # ✅ Return (image, success_flag)
        
#     except Exception as e:
#         print(f"Image generation error: {e}")
#         return None, False
    
