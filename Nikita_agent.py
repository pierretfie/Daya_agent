#!/usr/bin/env python3

import os
import shlex
from llama_cpp import Llama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
from datetime import datetime
from pathlib import Path
import re
import time
import sys
import warnings
import contextlib
import torch
from rich.console import Console

# Determine the directory of the main script (Nikita_agent.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base directory for Nikita (where model and output files are)
NIKITA_BASE_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model")

# Construct absolute paths relative to NIKITA_BASE_DIR
MODEL_PATH = os.path.join(NIKITA_BASE_DIR, "mistral.gguf")
OUTPUT_DIR = os.path.join(NIKITA_BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "history.json")
CHAT_HISTORY_FILE = Path(os.path.join(NIKITA_BASE_DIR, "nikita_history.json"))
COMMAND_HISTORY_FILE = os.path.join(NIKITA_BASE_DIR, "command_history")

# Construct absolute paths to scripts (in the same directory as Nikita_agent.py)
PROMPT_TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "modules", "prompt_template.txt")
FINE_TUNING_FILE = os.path.join(SCRIPT_DIR, "modules", "fine_tuning.json")

# Add the script directory to the Python path
sys.path.insert(0, SCRIPT_DIR)

console = Console()

from modules.intent_analyzer import IntentAnalyzer
from modules.resource_management import get_system_info, get_dynamic_params, optimize_memory_resources, optimize_cpu_usage, prewarm_model
from modules.history_manager import setup_command_history, save_command_history, get_input_with_history, load_chat_history, save_chat_history
from modules.context_optimizer import ContextOptimizer
from modules.semantic_context_optimizer import SemanticContextOptimizer
from modules.command_handler import run_command
from modules.engagement_manager import extract_targets, suggest_attack_plan, engagement_memory
from modules.reasoning_engine import ReasoningEngine
from modules.tool_manager import ToolManager
from modules.gpu_manager import GPUManager, is_gpu_available, get_gpu_memory




# Create necessary directories
os.makedirs(NIKITA_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import model parameters from context_optimizer
from modules.context_optimizer import DEFAULT_MAX_TOKENS, DEFAULT_RESERVE_TOKENS

# Model parameters
RESERVE_TOKENS = DEFAULT_RESERVE_TOKENS
MAX_TOKENS = DEFAULT_MAX_TOKENS  # Use the same value as context_optimizer
TEMPERATURE = 0.5  # Reduced from 0.7 for more focused responses
# Maximum number of messages to keep in memory
MEMORY_LIMIT = 20  # Set a reasonable limit for memory usage

# Get system parameters for model initialization
system_params = get_dynamic_params()

# Global variables for commands
system_commands = {}
categorized_commands = {}

# ===============================
# === COMMAND HISTORY ===========
# ===============================

def discover_system_commands():
    """Discover available system commands"""
    global system_commands, categorized_commands
    
    commands = {}
    categorized = {}
    
    # Common security tools
    security_tools = {
        "nmap": "Network scanner",
        "metasploit": "Exploitation framework",
        "hydra": "Login brute forcer",
        "hashcat": "Password cracker",
        "gobuster": "Directory bruteforcer",
        "wireshark": "Network analyzer",
        "aircrack-ng": "WiFi security tool",
        "burpsuite": "Web security tool",
        "sqlmap": "SQL injection tool"
    }
    
    # Populate some basic commands
    base_commands = {
        "ls": "List directory contents",
        "cd": "Change directory",
        "cat": "Display file contents",
        "grep": "Search text",
        "find": "Find files",
        "ip": "Show IP information",
        "ps": "Process status",
        "kill": "Terminate processes"
    }
    
    commands.update(base_commands)
    commands.update(security_tools)
    
    # Categorize commands
    categorized["file"] = ["ls", "cat", "find"]
    categorized["network"] = ["nmap", "ip", "wireshark"]
    categorized["process"] = ["ps", "kill"]
    categorized["security"] = ["metasploit", "hydra", "hashcat", "gobuster", "aircrack-ng", "burpsuite", "sqlmap"]
    
    system_commands = commands
    categorized_commands = categorized
    
    return commands, categorized

# Initialize system commands
system_commands, categorized_commands = discover_system_commands()

# ===============================
# === FINE TUNING IMPLEMENTATION
# ===============================

class FinetuningKnowledge:
    def __init__(self):
        self.knowledge_base = {}
        self.categories = set()
        self.tools = set()
        self.load_knowledge()

    def load_knowledge(self):
        try:
            with open(FINE_TUNING_FILE, "r") as f:
                data = json.load(f)
                for entry in data:
                    # Create category index
                    category = entry["category"]
                    self.categories.add(category)
                    if category not in self.knowledge_base:
                        self.knowledge_base[category] = []
                    self.knowledge_base[category].append(entry)

                    # Track tools
                    self.tools.add(entry["tool_used"])

                # Don't display knowledge loading information to keep output clean
        except Exception as e:
            console.print(f"[red]Error loading fine-tuning data: {e}[/red]")
            self.knowledge_base = {}

    def get_command_for_task(self, task):
        """Get the most relevant command for a given task"""
        task_lower = task.lower()
        
        # Skip the matching for informational queries to prevent hallucination
        if task_lower.startswith("what is") or task_lower.startswith("tell me about") or task_lower.startswith("explain"):
            # These are likely informational queries, not tool/command requests
            return None
        
        best_match = None
        best_score = 0
        
        # Extract the main tool name if explicitly mentioned in the query
        explicitly_mentioned_tool = None
        for category, entries in self.knowledge_base.items():
            for entry in entries:
                tool = entry["tool_used"].lower()
                if tool != "none" and tool in task_lower:
                    # Check that the tool name is a standalone word, not part of another word
                    for word in task_lower.split():
                        if tool == word:
                            explicitly_mentioned_tool = tool
                            break
        
        # If a specific tool is explicitly mentioned, require entries to match that tool
        if explicitly_mentioned_tool:
            for category, entries in self.knowledge_base.items():
                for entry in entries:
                    if entry["tool_used"].lower() == explicitly_mentioned_tool:
                        score = 5  # Base score for exact tool match
                        
                        # Additional scoring based on instruction match
                        instruction_words = set(entry["instruction"].lower().split())
                        task_words = set(task_lower.split())
                        common_words = instruction_words & task_words
                        score += len(common_words)
                        
                        if score > best_score:
                            best_score = score
                            best_match = entry
            
            # Only return if we have a good confidence match
            if best_score >= 6:
                return best_match
            # If no good match found with explicit tool, return None to prevent hallucination
            return None
        
        # Fall back to general matching only if no explicit tool mentioned
        for category, entries in self.knowledge_base.items():
            for entry in entries:
                score = 0

                # Check instruction match
                instruction_words = set(entry["instruction"].lower().split())
                task_words = set(task_lower.split())
                common_words = instruction_words & task_words
                score += len(common_words) * 2

                # Check if key phrases match
                if "how to" in task_lower and "how to" in entry["instruction"].lower():
                    score += 2
                if entry["tool_used"].lower() in task_lower:
                    score += 3

                # Check category relevance
                if any(word in task_lower for word in category.lower().split()):
                    score += 2

                # Update best match if this score is higher
                if score > best_score:
                    best_score = score
                    best_match = entry

        # Return the best match if it has a high minimum score to avoid weak matches
        return best_match if best_score >= 5 else None

    def suggest_next_steps(self, current_task, output):
        """Suggest next steps based on current task and output"""
        suggestions = []
        current_category = None

        # Find the current category
        for cat, entries in self.knowledge_base.items():
            for entry in entries:
                if entry["instruction"].lower() in current_task.lower():
                    current_category = cat
                    break
            if current_category:
                break

        if current_category:
            # Look for related commands in the same category
            for entry in self.knowledge_base[current_category]:
                # Skip the current command
                if entry["instruction"].lower() in current_task.lower():
                    continue
                
                # Add related commands as suggestions
                suggestion = {
                    "command": entry["tool_used"],
                    "purpose": entry["instruction"],
                    "relevance": "Same category: " + current_category
                }
                suggestions.append(suggestion)

            # If we found error patterns in the output, suggest fixes
            error_patterns = {
                "permission denied": ["Try using sudo", "Check file permissions"],
                "command not found": ["Install the required tool", "Check PATH environment"],
                "no such file": ["Verify the file path", "Create the directory first"],
                "connection refused": ["Check if service is running", "Verify port number"],
                "timeout": ["Check network connectivity", "Increase timeout value"]
            }

            output_lower = output.lower() if output else ""
            for pattern, fixes in error_patterns.items():
                if pattern in output_lower:
                    for fix in fixes:
                        suggestions.append({
                            "command": None,
                            "purpose": fix,
                            "relevance": "Error fix: " + pattern
                        })

        # Add general next steps based on common workflows
        common_workflows = {
            "scan": ["Analyze scan results", "Target specific ports", "Run deeper scan"],
            "exploit": ["Verify vulnerability", "Check for patches", "Document findings"],
            "enumerate": ["Filter results", "Identify critical assets", "Map network"],
            "brute": ["Review found credentials", "Test access", "Strengthen wordlist"]
        }

        for keyword, steps in common_workflows.items():
            if keyword in current_task.lower():
                for step in steps:
                    suggestions.append({
                        "command": None,
                        "purpose": step,
                        "relevance": "Workflow: " + keyword
                    })

        return suggestions

# ===============================
# === REASONING ENGINE =========
# ===============================

# Initialize reasoning engine
reasoning_engine = ReasoningEngine()

# ===============================
# === MEMORY ENGINE =============
# ===============================

# Memory for storing information
memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "system_info": {}
}

# Chat history
chat_memory = []

# Engagement memory
engagement_memory = {
    "targets": [],
    "credentials": [],
    "loot": [],
    "network_maps": [],
    "attack_history": []
}

# ===============================
# === CONTEXT OPTIMIZER ========
# ===============================

# No need to redefine ContextOptimizer class since we're importing it
# from modules.context_optimizer

# ===============================
# === Model Setup ===
console.print("🧠 [bold red]Waking Nikita 🐺...[/bold red]")

# Optimize system resources
success, aggressive_mode = optimize_memory_resources()

# Get system parameters
system_params = get_dynamic_params()

# Memory usage statistics - using simplified format
ram, swap, cpu_count, ram_gb = get_system_info()
console.print(f"[green]⚙️ RAM Tier: {ram_gb:.1f}GB system | Using {int(system_params['memory_target_gb'])}GB | Target: {system_params['memory_target_pct']:.1f}%[/green]")
console.print(f"[green]📊 Current usage: {ram.used/1024/1024/1024:.1f}GB ({ram.percent:.1f}%) | Context: {system_params['context_limit']} tokens | Batch: {system_params['n_batch']}[/green]")

# Display memory optimization status if used aggressive mode
if aggressive_mode:
    console.print("💫 [green]Aggressive memory optimization activated[/green] ")

# Apply CPU optimizations
success, target_cores, current_load = optimize_cpu_usage()
console.print(f"[green]⚡ CPU affinity set to use {target_cores} cores based on current load ({current_load:.2f})[/green]  ")

# Ensure mlock is enabled for RAM optimization
try:
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
except:
    pass

# === GPU Power Check Function ===
def is_gpu_powerful(device_info):
    """Determine if a GPU is powerful based on its specifications"""
    if not device_info:
        return False
        
    # Memory check (in bytes)
    memory_gb = device_info['global_mem_size'] / (1024**3)
    memory_powerful = memory_gb >= 8  # 8GB or more is considered powerful
    
    # Compute units check
    compute_units = device_info['max_compute_units']
    compute_powerful = compute_units >= 16  # 16 or more compute units is powerful
    
    # Work group size check
    work_group_size = device_info['max_work_group_size']
    work_group_powerful = work_group_size >= 256  # 256 or more is powerful
    
    # Overall assessment
    is_powerful = (
        memory_powerful and 
        compute_powerful and 
        work_group_powerful
    )
    
    return {
        'is_powerful': is_powerful,
        'memory_gb': memory_gb,
        'compute_units': compute_units,
        'work_group_size': work_group_size,
        'details': {
            'memory_powerful': memory_powerful,
            'compute_powerful': compute_powerful,
            'work_group_powerful': work_group_powerful
        }
    }

# === Load Prompt Template ===
try:
    with open(PROMPT_TEMPLATE_FILE, "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    console.print(f"[red]Error:[/red] Prompt template file not found at {PROMPT_TEMPLATE_FILE}")
    PROMPT_TEMPLATE = "You are a helpful AI assistant."  # Default prompt
except Exception as e:
    console.print(f"[yellow]Could not load prompt template: {e}[/yellow]")
    PROMPT_TEMPLATE = "You are a helpful AI assistant."  # Default prompt

# Create a context manager to redirect stderr
@staticmethod
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Set environment variables to suppress GPU and model logs
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Suppress CUDA initialization messages
os.environ['LLAMA_CPP_LOG_LEVEL'] = '-1'  # Suppress llama.cpp logs
os.environ['LLAMA_CPP_VERBOSE'] = '0'  # Suppress additional llama.cpp verbose output

# Initialize model with stderr suppression
llm = None # Initialize llm to None

# Start stderr suppression before any imports or initialization
with suppress_stderr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Initialize GPU manager with minimal output
            gpu_manager = GPUManager()
            gpu_manager.set_suppress_output(True)  # Suppress GPU manager logs
            
            # Redirect stderr during GPU initialization
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stderr(fnull):
                    # Initialize GPU manager (can take args like preferred_gpu='nvidia')
                    gpu_init_success = gpu_manager.initialize()
            
            device_info = None # Default to None
            n_gpu_layers = 0   # Default to CPU-only

            if gpu_init_success:
                device_info = gpu_manager.get_device_info()
            
            # Check if device_info is valid before accessing keys
            if device_info:
                power_analysis = is_gpu_powerful(device_info) # power_analysis is defined above
                
                # Use appropriate settings based on power analysis (example, adjust as needed)
                if power_analysis['is_powerful']:
                    work_split_ratio = 0.7  # Example: More GPU
                    # tensor_split = [0.3, 0.7] # tensor_split isn't directly used by Llama constructor here
                else:
                    work_split_ratio = 0.3  # Example: More CPU
                    # tensor_split = [0.7, 0.3]

                # Print minimal GPU info
                mem_gb = device_info.get('global_mem_size', 0) / (1024**3)
                console.print(f"[cyan]GPU:[/cyan] [green]{device_info.get('name', 'N/A')} ({mem_gb:.1f} GB)[/green]")
                
                # --- CORRECTED SECTION ---
                # Set n_gpu_layers based on device source and llama compatibility
                if device_info.get('source').lower() == 'cuda' and device_info.get('llama_compatible', False):
                    # Use GPU for compatible CUDA devices
                    assigned_layers = device_info.get('llama_layers_assigned', 0)
                    console.print(f"[green]GPU Acceleration: {assigned_layers if assigned_layers != 0 else 'All'} layers[/green]")
                    n_gpu_layers = assigned_layers # This will be -1 for "all" or a specific number
                else:
                    console.print("[yellow]Using CPU only[/yellow]")
                    n_gpu_layers = 0
                    
            else:
                # Handle case where GPU manager init failed or no device found
                console.print("[yellow]GPU Manager initialization failed or no device found. Using CPU-only mode.[/yellow]")
                # Set default splits if device info is unavailable (less relevant now)
                work_split_ratio = 0.0 # No GPU work
                n_gpu_layers = 0

            # Initialize Llama model with verbose=False to minimize logging
            # Set environment variable to suppress llama.cpp logs
            os.environ['LLAMA_CPP_LOG_LEVEL'] = '-1'
            
            # Initialize Llama with minimal logging
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=system_params['context_limit'],
                n_threads=system_params['n_threads'], # Use calculated threads
                n_batch=system_params['n_batch'],     # Use calculated batch size
                use_mlock=True,                       # Keep True for performance if RAM allows
                use_mmap=True,                        # Generally good for loading
                # low_vram=True, # Only enable if truly low VRAM, can impact performance
                verbose=False,                        # Ensure verbose is False
                # f16_kv=True, # Keep True if model supports and helps performance/memory
                seed=42,
                embedding=False, # Likely False for chat models
                # rope_scaling={"type": "linear", "factor": 0.25}, # Only if needed for context > trained length
                n_gpu_layers=n_gpu_layers,            # Use calculated layers
                # vocab_only=False, # Keep False for generation
                # main_gpu=0, # Let llama.cpp decide if n_gpu_layers > 0
                # tensor_split=None, # Usually let llama.cpp handle this based on n_gpu_layers
                # gpu_memory_utilization=0.8 if n_gpu_layers != 0 else 0.0, # Let llama.cpp manage if possible
                # logits_all=False, # Keep False unless needed for specific analysis
                # last_n_tokens_size=64, # Default is usually fine
                # cache=True # Llama.cpp manages internal caching
            )
            
            # Prewarm the model AFTER successful initialization
            console.print("[cyan]🔥 Prewarming model...[/cyan]")
            prewarm_duration = prewarm_model(llm, base_prompt="You are Nikita, an AI Security Assistant.")
            console.print(f"✅ [green] Model prewarmed in {prewarm_duration:.2f} seconds[/green]")
            
            # Verify GPU usage without printing detailed logs
            if n_gpu_layers != 0: # Check if we intended to use GPU layers
                try:
                    if torch.cuda.is_available():
                        # Small delay to allow memory allocation to potentially settle
                        time.sleep(0.2) 
                        mem_allocated = torch.cuda.memory_allocated(0) / (1024**2)
                        console.print(mem_allocated)
                        # mem_reserved = torch.cuda.memory_reserved(0) / (1024**2) # Reserved is less indicative of active layers
                        if mem_allocated > 6: # Check if *some* significant memory is allocated
                            console.print("[green]✅ GPU appears active for inference (memory allocated).[/green]")
                        else:
                            console.print("[yellow]⚠️ GPU acceleration intended, but low memory allocated. Check model/driver compatibility.[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Could not verify GPU memory usage: {e}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error initializing model or GPU manager: {str(e)}[/red]")
            # Use traceback to get more detail if needed:
            # import traceback
            # traceback.print_exc() 
            if 'gpu_manager' in locals() and gpu_manager is not None and gpu_manager.is_initialized():
                 gpu_manager.cleanup()
            sys.exit(1)

# Initialize model cache
MODEL_CACHE = {}

# Initialize fine tuning knowledge
fine_tuning = FinetuningKnowledge()

# Initialize system commands
system_commands, categorized_commands = discover_system_commands()

# Initialize intent analyzer
intent_analyzer = IntentAnalyzer(OUTPUT_DIR, system_commands)

# Initialize context optimizers - ensure llm exists if needed, or pass None/handle later
base_context_optimizer = ContextOptimizer(
    max_tokens= DEFAULT_MAX_TOKENS,
    reserve_tokens= DEFAULT_RESERVE_TOKENS # Assuming reserve_tokens is independent of llm
)

# Initialize semantic context optimizer with base optimizer
context_optimizer = SemanticContextOptimizer(base_optimizer=base_context_optimizer)

# Define a function to get responses with optimized caching
def get_cached_response(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, prompt_type='full'):
    """Get response from model with optimized caching and error handling"""
    try:
        # Use the prompt as is for 'basic' type
        if prompt_type == 'basic':
            final_prompt = prompt
        # Use full prompt as default
        else:
            final_prompt = prompt

        # Check cache first
        cache_key = f"{final_prompt}_{max_tokens}_{temperature}"
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        # Check GPU memory before inference
        gpu_mem_before = 0
        if torch.cuda.is_available():
            try:
                gpu_mem_before = torch.cuda.memory_allocated(0) / (1024**2)  # in MB
            except:
                pass                

        # Generate new response with optimized settings
        try:
            output = llm(final_prompt, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["User:", "\nUser:", "USER:"],
                        echo=False,  # Disable echo for faster response
                        stream=False)  # Disable streaming for faster response
            
            # Check GPU memory after inference to verify GPU usage
            if torch.cuda.is_available():
                try:
                    gpu_mem_after = torch.cuda.memory_allocated(0) / (1024**2)  # in MB
                    gpu_mem_diff = gpu_mem_after - gpu_mem_before
                    # Only print for significant changes to avoid log spam
                    if gpu_mem_diff > 50 or gpu_mem_diff < -50:  # If memory changed by more than 50MB
                        console.print(f"[cyan]GPU Memory delta during inference: {gpu_mem_diff:.1f}MB[/cyan]")
                except:
                    pass
            
            # Cache the response
            if len(MODEL_CACHE) > 20:  # Limit cache size
                MODEL_CACHE.clear()
            MODEL_CACHE[cache_key] = output
            
            return output
        except Exception as e:
            console.print(f"[yellow]Model inference error: {str(e)}[/yellow]")
            return {"choices": [{"text": "I apologize, but I encountered an error processing your request."}]}
    except Exception as e:
        console.print(f"[yellow]Error: {str(e)}[/yellow]")
        return {"choices": [{"text": "I apologize, but I encountered an error processing your request."}]}

# Helper function for command confirmation and execution
def confirm_and_run_command(cmd):
    """Displays the command and asks for user confirmation before running."""
    if not cmd:
        console.print("[yellow]Attempted to run an empty command.[/yellow]")
        return

    console.print(f"\n[bold yellow]Proposed Command:[/bold yellow] [cyan]{cmd}[/cyan]")
    try:
        confirm = input("Execute this command? (Y/N): ").strip().lower()
        if confirm == 'y' or confirm == 'yes':
            console.print(f"[green]Executing command...[/green]")
            output = run_command(cmd) # Use the existing run_command function
            
            # Get suggestions from fine-tuning knowledge base
            if output:
                suggestions = fine_tuning.suggest_next_steps(cmd, output)
                if suggestions:
                    console.print("\n[bold cyan]Suggested next steps:[/bold cyan]")
                    for suggestion in suggestions:
                        if suggestion['command']:
                            console.print(f"[yellow]• {suggestion['purpose']}[/yellow] using [green]{suggestion['command']}[/green] ({suggestion['relevance']})")
                        else:
                            console.print(f"[yellow]• {suggestion['purpose']}[/yellow] ({suggestion['relevance']})")
                    console.print()
        else:
            console.print("[yellow]Command execution skipped by user.[/yellow]")
    except EOFError:
        console.print("\n[yellow]Input stream closed. Command execution skipped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during command confirmation: {e}. Command execution skipped.[/red]")

# === UTILITY FUNCTIONS ===

# Note: The following functions have been moved to modules.engagement_manager
# - extract_targets
# - suggest_attack_plan

# Note: The following functions have been moved to modules.history_manager
# - load_chat_history
# - save_chat_history

# Note: The following functions have been moved to modules.command_handler
# - save_command_output
# - run_command # We keep importing this as the confirmation helper calls it

# === REPLACE main() FUNCTION ===
def main():
    """Main function to run the Nikita agent"""
    global chat_memory, llm, gpu_manager # Ensure llm and gpu_manager are accessible

    # Setup command history with readline
    history_enabled = setup_command_history()

    

    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        console.print(f"\n[bold red]Error:[/bold red] Model file not found at {MODEL_PATH}")
        console.print("[yellow]Please ensure the model file is placed in the correct location.[/yellow]")
        return
        
    # Check if llm was successfully initialized before proceeding
    if llm is None:
        console.print("[red]Model initialization failed. Exiting.[/red]")
        # Attempt cleanup even if init failed
        if 'gpu_manager' in globals() and gpu_manager is not None:
             gpu_manager.cleanup()
        return # Exit main function gracefully

    chat_memory = load_chat_history(memory_limit=MEMORY_LIMIT, chat_history_file=CHAT_HISTORY_FILE)

    # Print version banner
    console.print("\n[bold red]╔══════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║[/bold red]     [bold white]NIKITA 🐺 AI AGENT v1.0[/bold white]      [bold red]║[/bold red]")
    console.print("[bold red]╚══════════════════════════════════════╝[/bold red]\n")

    # Import threading capabilities
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Model prewarming is now done during initialization phase above
    # prewarm_duration = prewarm_model(llm, base_prompt="You are Nikita, an AI Security Assistant.")
    # console.print(f"✅ [cyan] Model prewarmed in {prewarm_duration:.2f} seconds[/cyan]")
    
    console.print("✅ [cyan]Nikita (Offline Operator Mode) Loaded[/cyan]")
    console.print("\nType 'exit' to quit, or 'clear' to delete chat memory.")
    console.print("[cyan]Available prompt modes:[/cyan]")
    console.print("• [yellow]basic[/yellow] <command> - Basic mode with minimal context")
    console.print("• Regular commands use full context by default\n")
    if chat_memory:
        pass
        #console.print(f"💬 Loaded {len(chat_memory)} previous chat messages")
    
    # More concise display of active targets if any
    if engagement_memory["targets"]:
        console.print(f"🎯 Active targets: {', '.join(engagement_memory['targets'])}")

    # Create a thread pool for background tasks
    executor = ThreadPoolExecutor(max_workers=2)

    # Prefetch common prompts to warm up cache
    def prefetch_common_tasks():
        """Prefetch common prompts with better error handling"""
        try:
            # Reduce number of prefetch prompts
            simple_prompts = [
                "You are Nikita, an AI Security Assistant. How can I help?",
                "You are Nikita, an AI Security Assistant. Respond briefly.",
                "You are Nikita, an AI Security Assistant. Analyze this security concern."
            ]
            
            for prompt in simple_prompts:
                try:
                    # Add delay between prefetches
                    time.sleep(0.5)
                    # Run inference with minimal tokens in background
                    _ = llm(prompt, max_tokens=1)
                except Exception as e:
                    console.print(f"[yellow]Prefetch warning for prompt: {str(e)}[/yellow]")
                    continue
            return True
        except Exception as e:
            console.print(f"[yellow]Prefetch error: {str(e)}[/yellow]")
            return False

    # Start prefetching in background with better error handling
    try:
        prefetch_future = executor.submit(prefetch_common_tasks)
        
        # Let prefetch run without timeout
        try:
            prefetch_success = prefetch_future.result()  # No timeout
            if prefetch_success:
                console.print("[cyan]✅ Common prompts prefetched successfully[/cyan]")
            else:
                console.print("[yellow]⚠️ Some prompts failed to prefetch[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Prefetch error: {str(e)}[/yellow]")
            # Cancel the prefetch if it's still running
            if not prefetch_future.done():
                prefetch_future.cancel()
    except Exception as e:
        console.print(f"[yellow]Failed to start prefetch: {str(e)}[/yellow]")

    # Initialize tool manager after other initializations
    tool_manager = ToolManager(fine_tuning_file=FINE_TUNING_FILE)

    while True:
        try:
            console.print("\n[bold cyan]┌──(SUDO)[/bold cyan]")
            console.print(f"[bold cyan]└─>[/bold cyan] ", end="") 

            # Get user input with readline support (history, editing)
            if history_enabled:
                user_input = get_input_with_history()
            else:
                user_input = input().strip()

            # Handle empty input by continuing to the next loop iteration
            if not user_input:
                continue

            # Handle special commands
            user_input_lower = user_input.lower()
            if user_input_lower in ["exit", "quit"]:
                save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                if history_enabled:
                    save_command_history()
                console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                break
            elif user_input_lower == "clear":
                chat_memory.clear()
                console.print("[green]Chat memory has been cleared! 🧹[/green]")
                continue

            # Add user input to chat memory
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })

            # Enforce memory limit
            if len(chat_memory) > MEMORY_LIMIT:
                chat_memory = chat_memory[-MEMORY_LIMIT:]

            # Parallelize non-critical tasks
            def background_processing():
                try:
                    extract_targets(user_input)
                    return suggest_attack_plan(user_input)
                except Exception as e:
                    return None

            # Start background processing
            attack_plan_future = executor.submit(background_processing)

            # NEW: Analyze intent first (this is faster)
            intent_analysis = intent_analyzer.analyze(user_input)

            # Get attack plan result if ready, otherwise continue without waiting
            try:
                attack_plan = attack_plan_future.result(timeout=0.1)  # Short timeout
                if attack_plan:
                    console.print(f"\n[yellow][Attack Plan][/yellow] {attack_plan}")
            except:
                # Continue without attack plan if it's taking too long
                pass

            # Generate reasoning - do quickly and in background if possible
            try:
                # Pass the result of intent analysis to the reasoning engine
                reasoning_future = executor.submit(reasoning_engine.analyze_task, user_input, intent_analysis=intent_analysis)

                # Get reasoning result with timeout and better error handling
                try:
                    reasoning_result = reasoning_future.result(timeout=0.5)  # Short timeout to avoid blocking
                except Exception as e:
                    console.print(f"[yellow]Reasoning timeout or error: {str(e)}[/yellow]")
                    reasoning_result = {"reasoning": {}, "follow_up_questions": []}
                    if not reasoning_future.done():
                        reasoning_future.cancel()
            except Exception as e:
                console.print(f"[yellow]Failed to start reasoning: {str(e)}[/yellow]")
                reasoning_result = {"reasoning": {}, "follow_up_questions": []}

            # Generate base prompt
            base_prompt = context_optimizer.get_optimized_prompt(
                chat_memory=chat_memory[-10:],
                current_task=user_input,
                base_prompt=PROMPT_TEMPLATE if not user_input.lower().startswith("basic") else None,
                reasoning_context=reasoning_result.get("reasoning", {}),
                follow_up_questions=reasoning_result.get("follow_up_questions", []),
                tool_context=None if user_input.lower().startswith("basic") else 
                    tool_manager.get_tool_context(reasoning_result.get("tool_name")) if reasoning_result.get("tool_name") else None
            )
            
            # Determine prompt type based on user input
            prompt_type = 'full'  # default
            if user_input.lower().startswith("basic"):
                prompt_type = 'basic'

            # --- DEBUG START ---
            # print(f"\n{'='*20} DEBUG INFO {'='*20}")
            # print(f"Intent Analysis: {intent_analysis}")
            # print(f"--- Full Prompt Sent to LLM: ---\n{full_prompt}")
            # print("---------------------------------")
            # # --- DEBUG END ---

            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                start_time = time.time()
                task_id = progress.add_task("🐺 Reasoning...", total=None)
                
                # Update the timer every 0.1 seconds
                timer_running = True
                def update_timer():
                    while timer_running:
                        elapsed = time.time() - start_time
                        progress.update(task_id, description=f"🐺 Reasoning... [{elapsed:.1f}s]")
                        time.sleep(0.1)
                
                # Start the timer update thread
                timer_thread = threading.Thread(target=update_timer)
                timer_thread.daemon = True
                timer_thread.start()

                # Generate response using cached response function for better performance
                try:
                    output = get_cached_response(base_prompt, prompt_type=prompt_type)
                    response = output['choices'][0]['text'].strip()
                    
                    # Ensure we have a valid response
                    if not response or response.lower().startswith(('i apologize', 'sorry', 'error')):
                        # Generate a fallback response using the same prompt
                        output = get_cached_response(base_prompt, prompt_type=prompt_type)
                        response = output['choices'][0]['text'].strip()
                    
                    # Try to parse response as JSON first
                    try:
                        response_data = json.loads(response)
                        if isinstance(response_data, dict) and 'output_message' in response_data:
                            clean_response = response_data['output_message']
                        else:
                            clean_response = response
                    except json.JSONDecodeError:
                        # If not JSON, filter out reasoning sections and metadata
                        response_lines = response.split('\n')
                        filtered_response = []
                        skip_section = False
                        for line in response_lines:
                            line = line.strip()
                            # Skip JSON-like lines and reasoning sections
                            if line.startswith(('{', '}', 'Understanding:', 'Response:', '"input_context":', '"domain_context":', '"output_context":', '"output_format":', '"output_type":', '"output_level":', '"output_tone":', '"output_format_version":', 'Reasoning:', 'Follow-up Questions:', 'Task:')):
                                skip_section = True
                                continue
                            # Reset skip_section when we hit an empty line
                            if not line:
                                skip_section = False
                            # Only add lines when not in a skip section
                            if not skip_section and line:
                                filtered_response.append(line)
                        clean_response = '\n'.join(filtered_response).strip()
                    
                    # Ensure we have content to display
                    if not clean_response:
                        clean_response = "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."
                    
                    # Stop the timer thread and progress spinner before displaying response
                    timer_running = False
                    timer_thread.join(timeout=0.2)  # Wait for thread to finish
                    progress.stop()
                    
                    # Display total elapsed time
                    total_time = time.time() - start_time
                    console.print(f"⏱️ {total_time:.1f}s")
                    
                    # Display the response with clear formatting
                    console.print(f"\n[bold magenta]┌──(NIKITA 🐺)[/bold magenta]")
                    console.print(f"[bold magenta]└─>[/bold magenta] {clean_response}")
                    console.print() # Add an empty line after output for better readability
                    
                except Exception as e:
                    console.print(f"[red]Error generating response: {str(e)}[/red]")
                    console.print("[yellow]Please try rephrasing your question.[/yellow]")

                # Process any commands in the response
                command_match = re.search(r'```(.*?)```', response, re.DOTALL) or re.search(r'`(.*?)`', response, re.DOTALL)
                
                executed_command_this_turn = False # Flag to avoid double execution

                if intent_analysis and intent_analysis.get("command") and intent_analysis.get("should_execute", False):
                    # Execute command from intent analysis (with confirmation)
                    cmd = intent_analysis["command"]
                    
                    # Check if it's a help request (don't need confirmation for help)
                    if cmd.lower().startswith(('help', 'man')):
                        tool_name = cmd.split()[-1]
                        if tool_name in system_commands:
                            tool_help = tool_manager.get_tool_help(tool_name)
                            if tool_help:
                                console.print(f"\n[bold cyan]Help for {tool_name}:[/bold cyan]")
                                if tool_help.get("source") == "man_page":
                                    console.print(f"[bold]Name:[/bold] {tool_help.get('name', 'N/A')}")
                                    console.print(f"[bold]Synopsis:[/bold] {tool_help.get('synopsis', 'N/A')}")
                                    console.print(f"[bold]Description:[/bold] {tool_help.get('description', 'N/A')}")
                                    if tool_help.get('options'):
                                        console.print(f"[bold]Options:[/bold] {tool_help['options']}")
                                    if tool_help.get('examples'):
                                        console.print(f"[bold]Examples:[/bold] {tool_help['examples']}")
                                else:
                                    console.print(tool_help.get('help_text', 'No help text found.'))
                            else:
                                console.print(f"[yellow]No help information available for {tool_name}[/yellow]")
                            executed_command_this_turn = True # Treat help display as handled
                        else:
                            # If help is for an unknown command, ask to run it
                            confirm_and_run_command(cmd)
                            executed_command_this_turn = True
                    else:
                        # Ask for confirmation for non-help commands
                        confirm_and_run_command(cmd)
                        executed_command_this_turn = True

                elif command_match and not executed_command_this_turn:
                    # Execute command from response (with confirmation)
                    cmd = command_match.group(1).strip()
                    
                    # Check if it's a help request (don't need confirmation)
                    if cmd.lower().startswith(('help', 'man')):
                        tool_name = cmd.split()[-1]
                        if tool_name in system_commands:
                            tool_help = tool_manager.get_tool_help(tool_name)
                            if tool_help:
                                console.print(f"\n[bold cyan]Help for {tool_name}:[/bold cyan]")
                                if tool_help.get("source") == "man_page":
                                    console.print(f"[bold]Name:[/bold] {tool_help.get('name', 'N/A')}")
                                    console.print(f"[bold]Synopsis:[/bold] {tool_help.get('synopsis', 'N/A')}")
                                    console.print(f"[bold]Description:[/bold] {tool_help.get('description', 'N/A')}")
                                    if tool_help.get('options'):
                                        console.print(f"[bold]Options:[/bold] {tool_help['options']}")
                                    if tool_help.get('examples'):
                                        console.print(f"[bold]Examples:[/bold] {tool_help['examples']}")
                                else:
                                    console.print(tool_help.get('help_text', 'No help text found.'))
                            else:
                                console.print(f"[yellow]No help information available for {tool_name}[/yellow]")
                            executed_command_this_turn = True
                        else:
                            # If help is for an unknown command, ask to run it
                            confirm_and_run_command(cmd)
                    else:
                        # Ask for confirmation for non-help commands
                        confirm_and_run_command(cmd)
                
                # Save the response to chat memory in background
                def save_response_to_memory():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    chat_memory.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp,
                        "reasoning_context": reasoning_result.get("reasoning", {}),
                        "follow_up_questions": reasoning_result.get("follow_up_questions", [])
                    })
                    save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                
                # Execute save in background to avoid blocking
                executor.submit(save_response_to_memory)
        except KeyboardInterrupt:
            # Make sure to stop the timer if it exists in this scope
            if 'timer_running' in locals():
                timer_running = False
                if 'timer_thread' in locals():
                    timer_thread.join(timeout=0.2)
                # Display total elapsed time
                if 'start_time' in locals():
                    total_time = time.time() - start_time
                    console.print(f"⏱️ {total_time:.1f}s")
            console.print("[yellow]Processing interrupted by user[/yellow]")
            
            # Handle the interruption gracefully
            console.print("\n[yellow]Command interrupted. Press Ctrl+C again to exit or Enter to continue.[/yellow]")
            try:
                # Give the user a chance to exit with another Ctrl+C or continue
                if input().strip().lower() in ["exit", "quit"]:
                    save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                    if history_enabled:
                        save_command_history()
                    console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                    break
            except KeyboardInterrupt:
                save_chat_history(chat_memory, chat_history_file=CHAT_HISTORY_FILE)
                if history_enabled:
                    save_command_history()
                console.print("\n[bold red]Nikita:[/bold red] Exiting. Stay frosty.\n")
                break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {str(e)}")
            console.print("[yellow]Continuing to next prompt...[/yellow]")
    
    # Shutdown executor pool cleanly
    executor.shutdown(wait=False)

    # Explicitly clean up resources before exit
    print("Cleaning up resources...")
    # Try closing llama object first
    if 'llm' in globals() and llm is not None:
        try:
            print("Closing Llama model...")
            llm.close() # Call the library's close method
            print("Llama model closed.")
        except Exception as e:
            print(f"Error closing Llama model: {e}")
        llm = None # Ensure reference is gone
    
    # Clean up GPU manager after Llama
    if 'gpu_manager' in globals() and gpu_manager is not None:
        try:
            print("Cleaning up GPU Manager...")
            gpu_manager.cleanup()
            print("GPU Manager cleaned up.")
        except Exception as e:
            print(f"Error cleaning up GPU Manager: {e}")
        gpu_manager = None # Ensure reference is gone

    # Optional: Force garbage collection again after explicit cleanup
    import gc
    gc.collect()
    print("Cleanup complete.")

def select_device():
    if torch.cuda.is_available() and is_gpu_available():
        gpu_memory = get_gpu_memory()
        if gpu_memory > 2.0:
            return torch.device("cuda")
    return torch.device("cpu")

if __name__ == "__main__":
    main()