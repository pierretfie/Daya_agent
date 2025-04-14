import requests
import json
import os
import logging

# Assuming ContextOptimizer and ResponseCleaner are in the modules directory
# and FinetuningKnowledge is defined in Daya_agent.py (adjust path if needed)
try:
    from modules.context_optimizer import ContextOptimizer
    from modules.response_cleaner import ResponseCleaner
    # FinetuningKnowledge might be directly in Daya_agent.py or moved to a module
    # Adjust this import based on the actual location of FinetuningKnowledge
    # If it's in Daya_agent.py, this relative import might need adjustment
    # or Daya_agent.py needs to be structured as a package.
    # For now, assuming it might be moved or accessible this way:
    # from .Daya_agent import FinetuningKnowledge # Example if Daya_agent is part of a package
    # Or, if Daya_agent.py is just a script, direct import might be tricky.
    # Let's assume FinetuningKnowledge is not strictly needed *inside* this handler
    # for the initial API call, but the context optimizer uses its data.
    print("Successfully imported modules for Gemini Handler.")
except ImportError as e:
    print(f"Error importing modules for Gemini Handler: {e}. Ensure modules are accessible.")
    # Define dummy classes if imports fail, to allow basic structure loading
    class ContextOptimizer: pass
    class ResponseCleaner: pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.warning("GEMINI_API_KEY environment variable not set. Gemini API calls will fail.")
    # Consider raising an error or exiting if the API key is mandatory for this handler's use

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
# Using gemini-pro as a default, consider making this configurable (e.g., gemini-1.5-flash)

class GeminiHandler:
    def __init__(self, context_optimizer: ContextOptimizer, response_cleaner: ResponseCleaner):
        """
        Initializes the Gemini Handler.

        Args:
            context_optimizer: An instance of ContextOptimizer to generate prompts.
            response_cleaner: An instance of ResponseCleaner to process responses.
            # finetuning_knowledge: An instance of FinetuningKnowledge (if needed directly).
        """
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        self.context_optimizer = context_optimizer
        self.response_cleaner = response_cleaner
        # self.finetuning_knowledge = finetuning_knowledge # Store if needed
        self.headers = {"Content-Type": "application/json"}
        logging.info("Gemini Handler initialized.")

    def generate_response(self, chat_memory, current_task, base_prompt, **kwargs) -> str:
        """
        Generates a response using the Gemini API.

        Args:
            chat_memory: List of chat messages.
            current_task: The current user task/query.
            base_prompt: The base system prompt.
            **kwargs: Additional arguments potentially needed by context optimizer.

        Returns:
            The cleaned text response from the Gemini API, or an error message.
        """
        logging.info(f"Generating Gemini response for task: {current_task}")
        
        if not self.context_optimizer:
             logging.error("Context optimizer not available.")
             return "Error: Context optimizer not initialized."
        if not self.response_cleaner:
             logging.error("Response cleaner not available.")
             return "Error: Response cleaner not initialized."

        try:
            # 1. Generate the optimized prompt using Daya's context optimizer
            # Pass necessary args like reasoning_context, tool_context if available in kwargs
            prompt_text = self.context_optimizer.get_optimized_prompt(
                chat_memory=chat_memory,
                current_task=current_task,
                base_prompt=base_prompt,
                reasoning_context=kwargs.get('reasoning_context'),
                follow_up_questions=kwargs.get('follow_up_questions'),
                tool_context=kwargs.get('tool_context')
            )
            
            logging.debug(f"Generated prompt for Gemini:\n{prompt_text}")

            # 2. Construct the payload for the Gemini API
            # Gemini API expects a specific structure, often with history formatted differently.
            # We need to adapt the prompt_text or chat_memory.
            # Simple approach: Send the entire generated prompt as the user's last message.
            # More complex: Format chat_memory into Gemini's 'contents' structure.
            
            # Simple approach payload:
            payload = {
                "contents": [
                    {
                        "role": "user", # The prompt includes history, so treat it as one user turn
                        "parts": [{"text": prompt_text}]
                    }
                ],
                # Add generation config (optional, defaults are usually okay)
                "generationConfig": {
                    "temperature": 0.7, # Example temperature
                    "maxOutputTokens": 2048 # Example limit
                },
                # Add safety settings (optional, use defaults or customize)
                # "safetySettings": [ ... ] 
            }

            # 3. Call the Gemini API
            logging.info("Sending request to Gemini API...")
            response = requests.post(GEMINI_API_URL, headers=self.headers, data=json.dumps(payload), timeout=120) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # 4. Process the response
            response_data = response.json()
            logging.debug(f"Received response data from Gemini: {response_data}")

            candidates = response_data.get("candidates", [])
            if candidates:
                # Get text from the first candidate
                gemini_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                
                if not gemini_text and candidates[0].get("finishReason") == 'SAFETY':
                     logging.warning("Gemini response blocked due to safety settings.")
                     return "My response was blocked due to safety filters. Please try rephrasing your request."
                elif not gemini_text:
                    logging.warning("Gemini returned an empty response.")
                    return "(Gemini returned an empty response)"
                    
                logging.info("Received non-empty response from Gemini.")
                
                # 5. Clean the response using Daya's response cleaner
                cleaned_result = self.response_cleaner.clean_response(gemini_text)
                cleaned_text = self.response_cleaner.format_for_display(cleaned_result)
                
                logging.info("Response cleaned successfully.")
                return cleaned_text
            elif response_data.get("promptFeedback", {}).get("blockReason"):
                 block_reason = response_data["promptFeedback"]["blockReason"]
                 logging.warning(f"Prompt blocked by Gemini API due to: {block_reason}")
                 return f"My ability to respond was blocked due to safety filters ({block_reason}). Please try rephrasing your request."
            else:
                logging.warning("Gemini API returned no candidates.")
                return "(Gemini returned no response candidates)"

        except requests.exceptions.RequestException as e:
            logging.error(f"Gemini API request failed: {e}")
            error_details = ""
            if e.response is not None:
                try:
                    error_details = e.response.json()
                except json.JSONDecodeError:
                    error_details = e.response.text
            return f"Error communicating with Gemini API: {e}. Details: {error_details}"
        except Exception as e:
            logging.error(f"Error processing Gemini response: {e}", exc_info=True)
            return f"An unexpected error occurred while processing the Gemini response: {e}"

# Example usage (for testing purposes)
if __name__ == "__main__":
    print("Testing Gemini Handler...")
    # This requires ContextOptimizer and ResponseCleaner to be properly importable
    # and the API key to be set.
    
    # Mock objects if imports failed or for standalone testing
    class MockContextOptimizer:
        def get_optimized_prompt(self, **kwargs):
            print("Mock Context Optimizer generating prompt...")
            # Return a simple prompt for testing
            return f"Base Prompt: {kwargs.get('base_prompt')}\nCurrent Task: {kwargs.get('current_task')}\nHistory: {kwargs.get('chat_memory')}"

    class MockResponseCleaner:
        def clean_response(self, text):
            print(f"Mock Cleaner processing text: '{text[:50]}...'")
            # Simple cleaning: remove potential prefixes
            cleaned = re.sub(r'^(Gemini|Response|AI):\s*', '', text, flags=re.IGNORECASE)
            return {"text": cleaned.strip()}
        def format_for_display(self, cleaned_result):
            return cleaned_result.get("text", "")

    try:
        # Replace with actual instances if imports work
        mock_optimizer = MockContextOptimizer()
        mock_cleaner = MockResponseCleaner()
        
        if not API_KEY:
             print("Skipping live API test as GEMINI_API_KEY is not set.")
        else:
            handler = GeminiHandler(context_optimizer=mock_optimizer, response_cleaner=mock_cleaner)
            
            test_memory = [{"role": "user", "content": "Hello"}]
            test_task = "Tell me a joke"
            test_base_prompt = "You are a helpful assistant."
            
            response = handler.generate_response(test_memory, test_task, test_base_prompt)
            print("\nGemini Response:")
            print(response)
            
    except Exception as e:
        print(f"Error during testing: {e}") 