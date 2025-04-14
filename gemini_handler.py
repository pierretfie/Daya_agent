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

# --- NEW: Import TTS functions ---
# Assuming tts_module.py is accessible
try:
    from tts_module import synthesize_to_temp_file, play_audio_file
    TTS_ENABLED = True
except ImportError:
    console.print("[yellow]⚠️ Warning: tts_module not found or failed to import. TTS features will be disabled.[/yellow]")
    TTS_ENABLED = False
    # Define dummy functions if import fails to avoid NameError later
    def synthesize_to_temp_file(text):
        return None
    def play_audio_file(file_path):
        pass
# --- END NEW ---

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
        # Use only the last N messages to avoid overly large payloads
        history_limit = 10
        start_index = max(0, len(chat_history) - history_limit)
        for msg in chat_history[start_index:]:
            # Skip empty messages which can cause API errors
            content = msg.get("content", "")
            if not content:
                continue
            role = "user" if msg.get("role") == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [{"text": content}]
            })

        # The final user message should be the *actual* user input from the last turn,
        # not the entire constructed prompt.
        # Assuming the last message in chat_history IS the user's latest input.
        # This requires the main loop to add the user input BEFORE calling generate_response.
        # (This matches the current Daya_agent structure)

        # If the history is not empty and the last message was from the user, use it.
        # Otherwise, we might need to extract it from the prompt (less ideal).
        # Let's refine the logic: the last item in gemini_history should always be the user's input.
        # The system prompt/context needs to be handled separately.

        system_context = ""
        # Attempt to extract system context (base prompt) from the full prompt string
        user_input_marker = "Task: "
        task_index = prompt.rfind(user_input_marker)
        if task_index != -1:
            system_context = prompt[:task_index].strip()
            # The user query part is already the last item added to gemini_history
        else:
            # Fallback: Try to identify a common system prompt structure
            common_prompt_end = "Provide a complete, detailed response:"
            sys_prompt_index = prompt.find(common_prompt_end)
            if sys_prompt_index != -1:
                system_context = prompt[:sys_prompt_index + len(common_prompt_end)].strip()
            else:
                # If no clear separation, maybe the prompt IS the system context?
                # Or maybe there's no system context. Defaulting to empty.
                system_context = "You are Daya, an AI Security Assistant." # Default fallback

        payload = {
            "contents": gemini_history,
            "generationConfig": {
                "temperature": 0.6, # Slightly lower temperature for more factual security answers
                "maxOutputTokens": 4096, # Increase token limit for potentially detailed answers
            },
             "safetySettings": [
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 # Allow potentially dangerous content for security context, but monitor
                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
             ]
        }

        # Add system instruction if context exists and is not empty
        if system_context:
            payload["system_instruction"] = {"parts": [{"text": system_context}]}

        return payload

    def generate_response(self, prompt: str, chat_history: list) -> tuple[str, str | None]:
        """
        Generates a response using the Gemini API and optionally synthesizes audio.

        Args:
            prompt (str): The prepared prompt string (used primarily for context extraction).
            chat_history (list): The *raw* chat history list for payload preparation.

        Returns:
            tuple[str, str | None]: A tuple containing:
                - The cleaned response text from the Gemini model.
                - The path to the temporary audio file if TTS was successful, otherwise None.
        """
        audio_file_path = None # Initialize audio file path
        payload = self._prepare_gemini_payload(prompt, chat_history)

        # --- DEBUG --- print(f"Gemini Payload: {json.dumps(payload, indent=2)}")

        console.print("[yellow]⏳ Sending request to Gemini API...[/yellow]")
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()

            response_data = response.json()
            # --- DEBUG --- console.print(f"[grey]Gemini Response Data: {json.dumps(response_data, indent=2)}[/grey]")

            candidates = response_data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts", []):
                raw_text = candidates[0]["content"]["parts"][0].get("text", "").strip()
                console.print("[green]✓ Response received from Gemini API.[/green]")

                # Clean the response using Daya's cleaner
                cleaned_result = self.response_cleaner.clean_response(raw_text)
                final_text = cleaned_result.get("text", "Error: Could not clean response.")

                # Check for safety blocks or finish reasons before TTS
                prompt_feedback = response_data.get("promptFeedback")
                if prompt_feedback and prompt_feedback.get("blockReason"):
                    block_reason = prompt_feedback.get("blockReason")
                    safety_ratings = prompt_feedback.get("safetyRatings", [])
                    console.print(f"[bold red]⚠️ Gemini API blocked the prompt: {block_reason}[/bold red]")
                    console.print(f"[red]Safety Ratings: {safety_ratings}[/red]")
                    final_text = f"Response blocked by API due to safety concerns: {block_reason}. Please modify your query."
                    return final_text, None # No audio for blocked responses

                finish_reason = candidates[0].get("finishReason")
                if finish_reason and finish_reason != "STOP":
                     console.print(f"[yellow]⚠️ Gemini API finish reason: {finish_reason}[/yellow]")
                     if finish_reason == "MAX_TOKENS":
                         final_text += "\n[Note: Response may be truncated due to maximum token limit.]"
                     elif finish_reason == "SAFETY":
                         final_text = "Response stopped by API due to safety concerns. Please modify your query."
                         return final_text, None # No audio for safety stops
                     elif finish_reason == "RECITATION":
                          final_text = "Response stopped by API due to recitation concerns."
                          return final_text, None # No audio for recitation stops
                     # Add handling for other reasons if necessary

                # --- NEW: Synthesize and Play Audio ---
                if TTS_ENABLED and final_text and not final_text.startswith("Error:"):
                    console.print("[cyan]🔊 Synthesizing audio...[/cyan]")
                    temp_audio_file = synthesize_to_temp_file(final_text)
                    if temp_audio_file:
                        console.print(f"[green]✓ Audio synthesized to {temp_audio_file}[/green]")
                        # We return the path, the main loop will handle playback
                        audio_file_path = temp_audio_file
                    else:
                        console.print("[yellow]⚠️ Failed to synthesize audio.[/yellow]")
                # --- END NEW ---

                return final_text, audio_file_path # Return text and audio path

            elif response_data.get("promptFeedback", {}).get("blockReason"):
                 block_reason = response_data["promptFeedback"]["blockReason"]
                 safety_ratings = response_data["promptFeedback"].get("safetyRatings", [])
                 console.print(f"[bold red]⚠️ Gemini API blocked the prompt: {block_reason}[/bold red]")
                 console.print(f"[red]Safety Ratings: {safety_ratings}[/red]")
                 return f"Prompt blocked by API due to safety concerns: {block_reason}. Please modify your query.", None
            else:
                console.print("[yellow]Gemini API returned an empty response or unexpected format.[/yellow]")
                console.print(f"[grey]Response Data: {json.dumps(response_data, indent=2)}[/grey]")
                return "Error: Received an empty or unexpected response from the API.", None

        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]❌ Error connecting to Gemini API: {e}[/bold red]")
            return f"Error: Could not connect to the Gemini API ({e}).", None
        except json.JSONDecodeError as e:
             console.print(f"[bold red]❌ Error decoding Gemini API response: {e}[/bold red]")
             try:
                 console.print("Raw response text:", response.text)
             except NameError:
                 pass
             return "Error: Could not decode the API response.", None
        except Exception as e:
            console.print(f"[bold red]❌ An unexpected error occurred during Gemini API interaction: {e}[/bold red]")
            # Optional: Log the full traceback for debugging
            # import traceback
            # console.print(f"[grey]{traceback.format_exc()}[/grey]")
            return f"Error: An unexpected error occurred ({e}).", None

# Example Self-Test (Optional)
if __name__ == "__main__":
    console.print("[bold blue]Running Gemini Handler Self-Test...[/bold blue]")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        console.print("[bold yellow]⚠️ Warning: GEMINI_API_KEY environment variable not set. Self-test skipped.[/bold yellow]")
    elif not TTS_ENABLED:
        console.print("[bold yellow]⚠️ Warning: TTS module not available. Audio self-test skipped.[/bold yellow]")
    else:
        try:
            gemini_handler = GeminiHandler(api_key=api_key)
            test_history = [
                {"role": "user", "content": "What is nmap?"},
                {"role": "assistant", "content": "Nmap is a popular network scanning tool."},
                # Add the latest user query here for the _prepare_gemini_payload logic
                {"role": "user", "content": "How do I scan for open ports using nmap?"}
            ]
            # The prompt now primarily provides system context
            test_prompt = """
You are Daya 🐺, an AI Security Assistant specializing in cybersecurity. Provide clear, concise, and technically accurate responses. Explain security concepts and tool usage thoroughly.
            """

            response_text, audio_file = gemini_handler.generate_response(test_prompt, test_history)
            console.print("\n[bold magenta]Test Response Text:[/bold magenta]")
            console.print(response_text)

            if audio_file:
                console.print(f"\n[cyan]🎧 Playing synthesized audio from: {audio_file}[/cyan]")
                play_audio_file(audio_file)
                # Optionally remove the temp file after playing
                # try:
                #     os.remove(audio_file)
                # except OSError as e:
                #     console.print(f"[yellow]Warning: Could not remove temp audio file {audio_file}: {e}[/yellow]")
            else:
                console.print("\n[yellow]No audio file generated for this response.[/yellow]")

        except ValueError as e:
            console.print(f"[bold red]Self-test failed during initialization: {e}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]An error occurred during self-test: {e}[/bold red]")
            import traceback
            console.print(f"[grey]{traceback.format_exc()}[/grey]") 