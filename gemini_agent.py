import os
import sys
import json
import requests
import time
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from Daya_agent import (
    main as daya_main,
    FinetuningKnowledge,
    ReasoningEngine,
    ToolManager,
    IntentAnalyzer,
    ResponseCleaner,
    ContextOptimizer,
    SemanticContextOptimizer,
    get_system_info,
    get_dynamic_params,
    optimize_memory_resources,
    optimize_cpu_usage,
    run_command,
    confirm_and_run_command
)
from tts_module import synthesize_to_temp_file, play_audio_file

console = Console()

# Initialize components from Daya
fine_tuning = FinetuningKnowledge()
reasoning_engine = ReasoningEngine()
tool_manager = ToolManager()
intent_analyzer = IntentAnalyzer()
response_cleaner = ResponseCleaner()
base_context_optimizer = ContextOptimizer()
context_optimizer = SemanticContextOptimizer(base_optimizer=base_context_optimizer)

def get_gemini_response(user_input, chat_memory=None, reasoning_context=None):
    """Get response from Gemini API with enhanced context"""
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        console.print("[red]❌ Please set the GEMINI_API_KEY environment variable.[/red]")
        return None

    # Build context from chat history and reasoning
    context_parts = []
    if chat_memory:
        for msg in chat_memory[-5:]:  # Last 5 messages for context
            role = "user" if msg["role"] == "user" else "model"
            context_parts.append(f"{role}: {msg['content']}")
    
    if reasoning_context:
        context_parts.append(f"Reasoning: {json.dumps(reasoning_context)}")

    # Add system prompt with fine-tuning context
    system_prompt = "You are Daya, an AI Security Assistant. "
    if fine_tuning.knowledge_base:
        system_prompt += "You have access to security tools and knowledge. "
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Build the full context
    full_context = "\n".join(context_parts) if context_parts else ""
    if full_context:
        full_context = f"Previous context:\n{full_context}\n\nCurrent query: {user_input}"
    else:
        full_context = user_input

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{system_prompt}\n\n{full_context}"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            candidates = response.json().get("candidates", [])
            if candidates:
                text = candidates[0]["content"]["parts"][0].get("text", "").strip()
                return text
            return "(empty response)"
        else:
            console.print(f"[red]Error {response.status_code}[/red]")
            try:
                console.print("Details:", response.json())
            except:
                console.print(response.text)
            return None
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None

def process_gemini_response(user_input, response, chat_memory):
    """Process Gemini response with Daya's tools"""
    # Clean the response
    cleaned_result = response_cleaner.clean_response(response)
    clean_response = response_cleaner.format_for_display(cleaned_result)
    
    # Analyze intent
    intent_analysis = intent_analyzer.analyze(user_input)
    
    # Get reasoning
    reasoning_result = reasoning_engine.analyze_task(user_input, intent_analysis=intent_analysis)
    
    # Process commands
    extracted_commands = cleaned_result.get('commands', [])
    command_to_execute = extracted_commands[0] if extracted_commands else None
    
    if intent_analysis and intent_analysis.get("command") and intent_analysis.get("should_execute", False):
        cmd = intent_analysis["command"]
        confirm_and_run_command(cmd)
    elif command_to_execute:
        confirm_and_run_command(command_to_execute)
    
    return clean_response, reasoning_result

def main():
    """Main function to run the agent with model selection"""
    console.print("\n[bold cyan]Select AI Model:[/bold cyan]")
    console.print("1. Daya (Local Mistral-7B)")
    console.print("2. Gemini (Online API)")
    
    choice = Prompt.ask("Enter your choice", choices=["1", "2"], default="1")
    
    if choice == "1":
        # Run the original Daya agent
        daya_main()
    else:
        # Run Gemini agent with enhanced features
        console.print("\n[bold green]Gemini Terminal Chat (type 'exit' to quit)[/bold green]")
        
        # Initialize chat memory
        chat_memory = []
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue
            
            # Add user input to chat memory
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            # Show thinking progress
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                start_time = time.time()
                task_id = progress.add_task("🤔 Thinking...", total=None)
                
                # Get reasoning context
                intent_analysis = intent_analyzer.analyze(user_input)
                reasoning_result = reasoning_engine.analyze_task(user_input, intent_analysis=intent_analysis)
                
                # Get response from Gemini
                response = get_gemini_response(
                    user_input,
                    chat_memory=chat_memory,
                    reasoning_context=reasoning_result.get("reasoning", {})
                )
                
                if response:
                    # Process the response
                    clean_response, reasoning_result = process_gemini_response(
                        user_input, response, chat_memory
                    )
                    
                    # Display response
                    console.print(f"\n[bold blue]Gemini:[/bold blue] {clean_response}")
                    
                    # Add response to chat memory
                    chat_memory.append({
                        "role": "assistant",
                        "content": clean_response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reasoning_context": reasoning_result.get("reasoning", {}),
                        "follow_up_questions": reasoning_result.get("follow_up_questions", [])
                    })
                    
                    # Optional: Add text-to-speech
                    temp_file = synthesize_to_temp_file(clean_response)
                    if temp_file:
                        play_audio_file(temp_file)
                else:
                    console.print("[red]Failed to get response from Gemini[/red]")

if __name__ == "__main__":
    main() 