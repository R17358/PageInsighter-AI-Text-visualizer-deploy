import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("gemini_key")
gemini_model = os.getenv("gemini_model")


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(gemini_model)

def answer_query(request):
    """Answer a general query using Gemini"""
   
    try:
        answer = generate_answer(request)
        translation = None
        
        print(answer)
    except Exception as e:
        print(e)
    

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
        print(e)


answer_query("who are you")