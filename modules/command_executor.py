"""
Command execution and selection logic for Daya Agent
"""
from rich.console import Console
from modules import run_command

console = Console()

def terminal_command_selection(commands):
    """
    Presents a numbered menu in the terminal for the user to select a command.
    Returns the selected command or None if cancelled.
    """
    if not commands:
        return None
    if len(commands) == 1:
        return commands[0]
    console.print("\n[bold yellow]Multiple valid commands were found. Please select which one to execute:[/bold yellow]")
    for idx, cmd in enumerate(commands, 1):
        console.print(f"  [cyan]{idx}[/cyan]: {cmd}")
    while True:
        try:
            choice = input(f"Enter the number of the command to execute (1-{len(commands)}) or 'q' to cancel: ").strip()
            if choice.lower() == 'q':
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(commands):
                return commands[int(choice)-1]
            else:
                console.print("[red]Invalid selection. Please try again.[/red]")
        except (EOFError, KeyboardInterrupt):
            return None

def confirm_and_run_command(cmd):
    """Displays the command and asks for user confirmation before running."""
    if not cmd:
        console.print("[yellow]Attempted to run an empty command.[/yellow]")
        return
    console.print(f"\n[bold yellow]Proposed Command:[/bold yellow] [cyan]{cmd}[/cyan]")
    try:
        confirm = input("Execute this command? (Y/N): ").lower()
        if confirm == 'y' or confirm == 'yes':
            console.print(f"[green]Executing command...[/green]")
            output = run_command(cmd) # Use the existing run_command function
            if output is not None:
                console.print(f"[green]Command executed successfully.[/green]")
                console.print(output)
            else:
                console.print(f"[red]Command execution failed or returned no output.[/red]")
        else:
            console.print("[yellow]Command execution cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during command execution: {e}[/red]")

def execute_cmd(intent_analysis, cleaned_result, system_commands, tool_manager):
    executed_command_this_turn = False  # Flag to avoid double execution
    extracted_commands = cleaned_result.get('commands', [])
    # Helper: is a command incomplete or just a tool name?
    def is_incomplete(cmd):
        if not cmd:
            return True
        cmd = cmd.strip()
        return (
            len(cmd.split()) == 1 or
            '<' in cmd or '>' in cmd
        )
    # Filter out incomplete commands
    valid_commands = [c for c in extracted_commands if not is_incomplete(c)]
    # If none from cleaner, try intent_analysis
    if not valid_commands and intent_analysis.get("command") and not is_incomplete(intent_analysis["command"]):
        valid_commands = [intent_analysis["command"]]
    # Let user select if multiple, or auto if one, or skip if none
    cmd = terminal_command_selection(valid_commands)
    print(f"Selected command for execution: {cmd}")
    if cmd is None:
        console.print("[yellow]No valid command selected (all were incomplete, had placeholders, or selection cancelled). Skipping execution.[/yellow]")
        return
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
            return
        else:
            # If help is for an unknown command, ask to run it
            confirm_and_run_command(cmd)
            return
    # For all other commands, ask for confirmation
    confirm_and_run_command(cmd)
