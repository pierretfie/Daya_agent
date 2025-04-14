import os
import sys
import json
import requests
from rich.console import Console
from rich.prompt import Prompt
from Daya_agent import main as daya_main
from tts_module import synthesize_to_temp_file, play_audio_file

console = Console()

def get_gemini_response(user_input):
    """Get response from Gemini API"""
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        console.print("[red]❌ Please set the GEMINI_API_KEY environment variable.[/red]")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_input
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
        # Run Gemini agent
        console.print("\n[bold green]Gemini Terminal Chat (type 'exit' to quit)[/bold green]")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue
            
            response = get_gemini_response(user_input)
            if response:
                console.print(f"[bold blue]Gemini:[/bold blue] {response}")
                # Optional: Add text-to-speech
                temp_file = synthesize_to_temp_file(response)
                if temp_file:
                    play_audio_file(temp_file)
            else:
                console.print("[red]Failed to get response from Gemini[/red]")

if __name__ == "__main__":
    main() 