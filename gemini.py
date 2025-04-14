import os
import requests
import json
from typing import Optional

API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def get_gemini_response(prompt: str) -> str:
    """
    Get response from Gemini API
    
    Args:
        prompt (str): The prompt to send to Gemini
        
    Returns:
        str: The response from Gemini
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        return "❌ Please set the GEMINI_API_KEY environment variable."
    
    headers = {
        "Content-Type": "application/json",
    }
    
    params = {
        "key": API_KEY
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, params=params, json=data)
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and result["candidates"]:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "No response generated from Gemini API."
            
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}" 