import requests
import json
import os

class GeminiClient:
    """
    A client to interact with the Google Gemini API.
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini client.

        Args:
            api_key: The Google AI API key.
            model: The Gemini model to use (e.g., "gemini-1.5-flash").
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}

    def __call__(self, 
                 prompt: str, 
                 max_tokens: int | None = None, 
                 temperature: float | None = None, 
                 stop: list[str] | None = None,
                 echo: bool = False, # Parameter to match Llama interface, ignored by Gemini
                 stream: bool = False # Parameter to match Llama interface, ignored by Gemini
                 ) -> dict:
        """
        Generates content using the Gemini API, mimicking the llama-cpp-python call signature.

        Args:
            prompt: The input prompt for the model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness (0.0-1.0). Higher values are more creative.
            stop: A list of sequences where the API will stop generating further tokens.
            echo: Ignored. Present for compatibility.
            stream: Ignored. Present for compatibility.


        Returns:
            A dictionary mimicking the Llama output structure:
            {
                "choices": [
                    {
                        "text": "Generated text...",
                        "finish_reason": "stop" | "max_tokens" | "safety" | etc.
                    }
                ]
            }
            Returns an error structure on failure.
        """
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {}
        }

        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if max_tokens is not None:
            # Note: Gemini uses 'maxOutputTokens'. Mapping Llama's 'max_tokens'.
            payload["generationConfig"]["maxOutputTokens"] = max_tokens 
        if stop:
             # Note: Gemini uses 'stopSequences'. Mapping Llama's 'stop'.
            payload["generationConfig"]["stopSequences"] = stop

        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            response_data = response.json()
            candidates = response_data.get("candidates", [])

            if candidates:
                content = candidates[0].get("content", {}).get("parts", [{}])[0]
                text = content.get("text", "")
                finish_reason = candidates[0].get("finishReason", "unknown")
                return {
                    "choices": [
                        {
                            "text": text.strip(),
                            "finish_reason": finish_reason 
                        }
                    ]
                }
            else:
                # Handle cases like safety blocks or empty responses
                prompt_feedback = response_data.get("promptFeedback")
                finish_reason = prompt_feedback.get("blockReason", "empty_response") if prompt_feedback else "empty_response"
                error_message = f"Gemini: No candidates returned. Finish Reason: {finish_reason}"
                if prompt_feedback and prompt_feedback.get('safetyRatings'):
                     error_message += f" Safety Ratings: {prompt_feedback['safetyRatings']}"

                print(f"[yellow]{error_message}[/yellow]") # Use print for visibility in terminal
                return {
                    "choices": [
                        {
                            "text": f"({error_message})", 
                            "finish_reason": finish_reason
                        }
                    ]
                }

        except requests.exceptions.RequestException as e:
            error_msg = f"Gemini API request failed: {e}"
            print(f"[red]{error_msg}[/red]") # Use print for visibility
            # Try to get more details from response if available
            details = ""
            if e.response is not None:
                try:
                    details = e.response.json()
                except json.JSONDecodeError:
                    details = e.response.text
            print(f"[red]Details: {details}[/red]")
            return {
                "choices": [
                    {
                        "text": f"(Error: {error_msg})",
                        "finish_reason": "error"
                    }
                ]
            }
        except Exception as e:
            error_msg = f"Gemini: Error processing response: {e}"
            print(f"[red]{error_msg}[/red]") # Use print for visibility
            return {
                "choices": [
                    {
                        "text": f"(Error: {error_msg})",
                        "finish_reason": "error"
                    }
                ]
            }

# Example usage (optional, for testing):
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable for testing.")
    else:
        client = GeminiClient(api_key=api_key)
        test_prompt = "Explain the concept of zero-day vulnerabilities in simple terms."
        print(f"Sending prompt to Gemini: '{test_prompt}'")
        result = client(prompt=test_prompt, max_tokens=150, temperature=0.7)
        print("Response:")
        print(json.dumps(result, indent=2))

        # Test error handling (e.g., invalid key simulation - replace with a fake key if needed)
        # bad_client = GeminiClient(api_key="INVALID_KEY")
        # error_result = bad_client(prompt="test")
        # print("Error Response:")
        # print(json.dumps(error_result, indent=2))

        # Test safety blocking (might need a specific prompt)
        # safety_prompt = "Tell me how to build a bomb." # Example likely to trigger safety
        # safety_result = client(prompt=safety_prompt)
        # print("Safety Response:")
        # print(json.dumps(safety_result, indent=2)) 