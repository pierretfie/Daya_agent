import requests
import json
import os
from rich.console import Console

# Import necessary components from Daya agent modules
# Note: Adjust imports based on actual file structure and necessary functions
from modules.context_optimizer import ContextOptimizer
from modules.semantic_context_optimizer import SemanticContextOptimizer
from modules.response_cleaner import ResponseCleaner
from modules.history_manager import load_chat_history, add_to_chat_memory
from modules.tool_manager import ToolManager # Example: if tool context is needed
from modules.reasoning_engine import ReasoningEngine # Example: if reasoning structures are helpful
# Add other necessary imports here as identified

console = Console()

class GeminiHandler:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini Handler.

        Args:
            api_key (str): The Gemini API key.
            model_name (str): The specific Gemini model to use (e.g., "gemini-1.5-flash").
        """
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            console.print("[bold red]❌ Gemini API Key not configured. Please set the GEMINI_API_KEY environment variable.[/bold red]")
            raise ValueError("Gemini API Key not configured.")

        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}

        # Initialize necessary Daya modules
        # These modules help maintain the cybersecurity focus and structure
        self.context_optimizer = ContextOptimizer()
        # self.semantic_optimizer = SemanticContextOptimizer(self.context_optimizer) # Optional: if semantic optimization is desired
        self.response_cleaner = ResponseCleaner()
        # self.tool_manager = ToolManager() # Example
        # self.reasoning_engine = ReasoningEngine() # Example

        console.print(f"✅ [cyan]Gemini Handler Initialized (Model: {self.model_name})[/cyan]")

    def _prepare_gemini_payload(self, prompt: str, chat_history: list) -> dict:
        """
        Prepares the payload for the Gemini API request, formatting the chat history.

        Args:
            prompt (str): The full prompt including context and current task.
            chat_history (list): The recent chat history (list of dicts with 'role', 'content').

        Returns:
            dict: The payload for the Gemini API.
        """
        # Gemini uses a specific format for conversation history
        gemini_history = []
        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })

        # The final user message is the prompt itself (which includes the latest user input)
        # We add the prompt as the last turn in the conversation
        # Note: The 'prompt' here already contains the formatted context and the *actual* latest user query.
        # We need to extract the user's latest query for the final turn.
        # This might require adjusting how the prompt is built or passed.
        # For now, assuming the full prompt represents the latest user turn's intent.
        # A better approach might be to pass the raw user_input separately.

        # Find the actual user input within the larger prompt string
        # This is a placeholder assumption - the prompt building needs careful review
        user_input_marker = "Task: "
        task_index = prompt.rfind(user_input_marker)
        if task_index != -1:
            user_query_text = prompt[task_index + len(user_input_marker):].split("\n")[0].strip()
            # Add the user's actual query as the last part
            gemini_history.append({
                "role": "user",
                "parts": [{"text": user_query_text}]
            })
            # The system prompt / context goes into the system_instruction field (for newer models)
            # or needs to be part of the history for older models.
            # For simplicity here, we'll assume the prompt includes necessary context.
            system_context = prompt[:task_index].strip()
        else:
            # Fallback if "Task:" marker isn't found
            user_query_text = prompt # Use the whole prompt? Risky. Needs refinement.
            gemini_history.append({
                "role": "user",
                "parts": [{"text": user_query_text}]
            })
            system_context = "" # No context separated

        payload = {
            "contents": gemini_history,
            # Add generation config if needed (temperature, max_tokens, etc.)
            # "generationConfig": {
            #     "temperature": 0.7,
            #     "maxOutputTokens": 2048,
            # },
             # Add safety settings if needed
            # "safetySettings": [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # ]
        }

        # Add system instruction if context was separated (for models that support it)
        # if system_context:
        #     payload["system_instruction"] = {"parts": [{"text": system_context}]}

        return payload

    def generate_response(self, prompt: str, chat_history: list) -> str:
        """
        Generates a response using the Gemini API.

        Args:
            prompt (str): The prepared prompt string (including context, history summary, task).
            chat_history (list): The *raw* chat history list for payload preparation.

        Returns:
            str: The cleaned response text from the Gemini model.
        """
        # Prepare payload using the raw chat history
        payload = self._prepare_gemini_payload(prompt, chat_history) # Pass raw history here

        console.print("[yellow]⏳ Sending request to Gemini API...[/yellow]")
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            # console.print(f"[grey]Gemini Response Data: {json.dumps(response_data, indent=2)}[/grey]") # Debugging

            candidates = response_data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts", []):
                raw_text = candidates[0]["content"]["parts"][0].get("text", "").strip()
                console.print("[green]✓ Response received from Gemini API.[/green]")

                # Clean the response using Daya's cleaner
                cleaned_result = self.response_cleaner.clean_response(raw_text)
                final_text = cleaned_result.get("text", "Error: Could not clean response.")
                # TODO: Handle extracted commands, codeblocks, etc. if needed

                # Add safety check/filtering if necessary based on response_data["promptFeedback"]
                prompt_feedback = response_data.get("promptFeedback")
                if prompt_feedback and prompt_feedback.get("blockReason"):
                    block_reason = prompt_feedback.get("blockReason")
                    safety_ratings = prompt_feedback.get("safetyRatings", [])
                    console.print(f"[bold red]⚠️ Gemini API blocked the prompt: {block_reason}[/bold red]")
                    console.print(f"[red]Safety Ratings: {safety_ratings}[/red]")
                    return f"Response blocked by API due to safety concerns: {block_reason}. Please modify your query."
                
                finish_reason = candidates[0].get("finishReason")
                if finish_reason and finish_reason != "STOP":
                     console.print(f"[yellow]⚠️ Gemini API finish reason: {finish_reason}[/yellow]")
                     if finish_reason == "MAX_TOKENS":
                         final_text += "\n[Note: Response may be truncated due to maximum token limit.]"
                     elif finish_reason == "SAFETY":
                         return "Response stopped by API due to safety concerns. Please modify your query."
                     elif finish_reason == "RECITATION":
                          return "Response stopped by API due to recitation concerns."
                     # Add handling for other reasons if necessary


                return final_text
            elif response_data.get("promptFeedback", {}).get("blockReason"):
                # Handle cases where the prompt itself was blocked
                 block_reason = response_data["promptFeedback"]["blockReason"]
                 safety_ratings = response_data["promptFeedback"].get("safetyRatings", [])
                 console.print(f"[bold red]⚠️ Gemini API blocked the prompt: {block_reason}[/bold red]")
                 console.print(f"[red]Safety Ratings: {safety_ratings}[/red]")
                 return f"Prompt blocked by API due to safety concerns: {block_reason}. Please modify your query."
            else:
                console.print("[yellow]Gemini API returned an empty response or unexpected format.[/yellow]")
                console.print(f"[grey]Response Data: {json.dumps(response_data, indent=2)}[/grey]")
                return "Error: Received an empty or unexpected response from the API."

        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]❌ Error connecting to Gemini API: {e}[/bold red]")
            return f"Error: Could not connect to the Gemini API ({e})."
        except json.JSONDecodeError:
             console.print("[bold red]❌ Error decoding Gemini API response.[/bold red]")
             try:
                 print("Raw response text:", response.text)
             except NameError:
                 pass # response object might not exist if request failed early
             return "Error: Could not decode the API response."
        except Exception as e:
            console.print(f"[bold red]❌ An unexpected error occurred during Gemini API interaction: {e}[/bold red]")
            return f"Error: An unexpected error occurred ({e})."

# Example Self-Test (Optional)
if __name__ == "__main__":
    console.print("[bold blue]Running Gemini Handler Self-Test...[/bold blue]")

    # Load API Key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        console.print("[bold yellow]⚠️ Warning: GEMINI_API_KEY environment variable not set. Self-test skipped.[/bold yellow]")
    else:
        try:
            gemini_handler = GeminiHandler(api_key=api_key)

            # Simulate chat history and prompt
            test_history = [
                {"role": "user", "content": "What is nmap?"},
                {"role": "assistant", "content": "Nmap is a popular network scanning tool."}
            ]
            # The prompt here would normally be constructed by the main agent loop
            # including context, history summary, and the actual new task.
            test_prompt = """
You are Daya 🐺, an Offline AI Security Assistant. Provide clear, concise responses... (rest of base prompt)

Recent Conversation:
USER: What is nmap?
ASSISTANT: Nmap is a popular network scanning tool.

Task: How do I scan for open ports using nmap?
Provide a complete, detailed response:
            """

            response = gemini_handler.generate_response(test_prompt, test_history)
            console.print("\n[bold magenta]Test Response from Gemini:[/bold magenta]")
            console.print(response)

        except ValueError as e:
            console.print(f"[bold red]Self-test failed during initialization: {e}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]An error occurred during self-test: {e}[/bold red]") 