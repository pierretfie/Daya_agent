#!/usr/bin/env python3
"""
Command Handler Module for Nikita Agent

Functions for running system commands, hardening commands with security best practices,
and saving command outputs.
"""

import os
import subprocess
import shlex
import re
from datetime import datetime
import psutil
import json
from typing import Dict, List, Tuple, Optional
import logging

# Try to import rich for pretty output if available
try:
    from rich.console import Console
    from rich.prompt import Confirm
    console = Console()
except ImportError:
    # Fallback to simple print if rich is not available
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def confirm(self, prompt: str) -> bool:
            response = input(f"{prompt} (y/n): ").lower()
            return response in ['y', 'yes']
    console = FallbackConsole()

# Default output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Nikita_Agent_model", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Command risk levels and categories
RISK_LEVELS = {
    "LOW": 1,      # Basic system info, non-destructive
    "MEDIUM": 2,   # Network scanning, file operations
    "HIGH": 3,     # System modifications, security tools
    "CRITICAL": 4  # Potentially dangerous operations
}

COMMAND_CATEGORIES = {
    "system_info": {
        "risk_level": "LOW",
        "commands": ["uname", "ls", "ps", "df", "free", "top", "uptime"]
    },
    "network_scan": {
        "risk_level": "MEDIUM",
        "commands": ["nmap", "ping", "traceroute", "netstat", "ss"]
    },
    "security_tools": {
        "risk_level": "HIGH",
        "commands": ["sqlmap", "gobuster", "hashcat", "john", "hydra"]
    },
    "system_modification": {
        "risk_level": "CRITICAL",
        "commands": ["rm", "mv", "cp", "chmod", "chown", "useradd", "usermod"]
    }
}

def get_command_risk_level(cmd: str) -> Tuple[str, str]:
    """
    Determine the risk level and category of a command.
    
    Args:
        cmd (str): The command to analyze
        
    Returns:
        tuple: (risk_level, category)
    """
    cmd_base = cmd.split()[0].lower()
    
    # Check command categories
    for category, info in COMMAND_CATEGORIES.items():
        if cmd_base in info["commands"]:
            return info["risk_level"], category
            
    # Default to MEDIUM risk if unknown
    return "MEDIUM", "unknown"

def requires_confirmation(cmd: str) -> bool:
    """
    Check if a command requires user confirmation based on its risk level.
    
    Args:
        cmd (str): The command to check
        
    Returns:
        bool: True if confirmation is required
    """
    risk_level, _ = get_command_risk_level(cmd)
    return RISK_LEVELS[risk_level] >= RISK_LEVELS["MEDIUM"]

def get_confirmation_message(cmd: str, risk_level: str, category: str) -> str:
    """
    Generate a confirmation message for a command.
    
    Args:
        cmd (str): The command to confirm
        risk_level (str): The risk level of the command
        category (str): The category of the command
        
    Returns:
        str: Confirmation message
    """
    return f"""⚠️  Command requires confirmation:
Command: {cmd}
Risk Level: {risk_level}
Category: {category}

Type 'yes' to proceed or 'no' to cancel."""

def harden_command(cmd):
    """
    Enhance commands with smart defaults and output paths for better security and logging.
    
    Args:
        cmd (str): The command to harden
        
    Returns:
        str: The hardened command with additional safety parameters
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Tool-specific enhancements
    enhancements = {
        'nmap': [
            ('-sV', ' -sV'),  # Version detection
            ('-sC', ' -sC'),  # Default scripts
            ('-sn', ' -sn'),  # Ping scan
            ('-T4', ' -T4'),  # Timing template
            ('--stats-every', ' --stats-every 10s')  # Progress updates
        ],
        'sqlmap': [
            ('--batch', ' --batch'),  # Non-interactive mode
            ('--random-agent', ' --random-agent')  # Random user agent
        ],
        'gobuster': [
            ('-q', ' -q')  # Quiet mode
        ],
        'smbclient': [
            ('-N', '-N')  # No password prompt
        ],
        'dig': [
            ('+short', ' +short')  # Short output
        ],
        'hashcat': [
            ('--identify', ' --identify'),  # Hash type identification
            ('--quiet', ' --quiet'),  # Less verbose output
            ('--show', ' --show')  # Show cracked passwords
        ]
    }

    # Add output paths for tools that support it
    if cmd.startswith('nmap'):
        output_base = os.path.join(OUTPUT_DIR, f"nmap_scan_{timestamp}")
        if not any(opt in cmd for opt in ['-oA', '-oX', '-oN', '-oG', '-oS']):
            cmd += f" -oA {output_base}"
    elif cmd.startswith('sqlmap'):
        sql_output = os.path.join(OUTPUT_DIR, f"sqlmap_{timestamp}")
        os.makedirs(sql_output, exist_ok=True)
        if '--output-dir' not in cmd:
            cmd += f" --output-dir={sql_output}"
    elif cmd.startswith('gobuster'):
        if '-o' not in cmd:
            cmd += f" -o {os.path.join(OUTPUT_DIR, f'gobuster_{timestamp}.txt')}"

    # Fix common IP range syntax
    ip_range_match = re.search(r'(\d{1,3}\.){3}\d{1,3}(?:-\d{1,3})?', cmd)
    if ip_range_match:
        ip_range = ip_range_match.group(0)
        if '-' in ip_range:
            base_ip = ip_range.split('-')[0]
            cmd = cmd.replace(ip_range, base_ip + '/24')

    # Apply tool-specific enhancements
    for tool, rules in enhancements.items():
        if cmd.startswith(tool):
            for check, add in rules:
                if check not in cmd:
                    cmd += add

    return cmd

def save_command_output(cmd, output, error=None):
    """
    Save command output to a file for later reference.
    
    Args:
        cmd (str): The command that was executed
        output (str): Command stdout
        error (str, optional): Command stderr
        
    Returns:
        str: Path to the output file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"cmd_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"=== Command ===\n{cmd}\n\n")
        f.write(f"=== Output ===\n{output}\n")
        if error:
            f.write(f"\n=== Errors ===\n{error}\n")

    return output_file

def run_command(cmd, timeout=30):
    """Execute a system command and return the result"""
    logging.info(f"Executing command: {cmd}")
    
    # Check if this is a security-related command
    security_keywords = ['nmap', 'nikto', 'metasploit', 'msfconsole', 'exploit', 'vuln', 
                         'sqlmap', 'burp', 'hydra', 'hashcat', 'john', 'wireshark', 'tcpdump']
    is_security_cmd = any(keyword in cmd.lower() for keyword in security_keywords)
    
    # Increase timeout for security tools which may take longer
    if is_security_cmd:
        timeout = 60  # Security scans often take longer
    
    # Validate command safety
    if not is_command_safe(cmd):
        return {
            "success": False,
            "output": "This command has been blocked for security reasons.",
            "error": "Command rejected by security policy."
        }
    
    # Validate command format and arguments
    command, error = validate_command(cmd)
    if error:
        return {
            "success": False,
            "output": "",
            "error": error
        }
    
    # For security commands, add helpful context
    if is_security_cmd:
        if 'nmap' in cmd.lower():
            logging.info("Running Nmap scan - this may take some time depending on target scope")
        elif 'sqlmap' in cmd.lower():
            logging.info("Running SQLMap - this may take time to test for SQL injection vulnerabilities")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )
        
        # Use communicate with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        # Process returned different outputs
        if process.returncode == 0:
            result = {
                "success": True,
                "output": stdout,
                "error": ""
            }
            
            # For security commands, add parsed summary of results when possible
            if is_security_cmd:
                result["summary"] = parse_security_output(cmd, stdout)
        else:
            result = {
                "success": False,
                "output": stdout,
                "error": stderr
            }
        
        return result
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Command timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def parse_security_output(cmd, output):
    """Parse and summarize output from security tools"""
    summary = ""
    
    if 'nmap' in cmd.lower():
        # Extract open ports and services
        open_ports = re.findall(r'(\d+/\w+)\s+open\s+(\S+)', output)
        if open_ports:
            summary += "Open ports found:\n"
            for port in open_ports[:10]:  # Limit to first 10 for brevity
                summary += f"- {port[0]}: {port[1]}\n"
            if len(open_ports) > 10:
                summary += f"... and {len(open_ports)-10} more\n"
    
    elif 'sqlmap' in cmd.lower():
        # Extract SQLi vulnerabilities found
        if "is vulnerable" in output:
            vulns = re.findall(r'Parameter \'([^\']+)\' is vulnerable', output)
            if vulns:
                summary += "SQL Injection vulnerabilities found in parameters:\n"
                for vuln in vulns:
                    summary += f"- {vuln}\n"
        elif "all tested parameters do not appear to be injectable" in output:
            summary += "No SQL Injection vulnerabilities were found.\n"
    
    elif 'gobuster' in cmd.lower():
        # Extract directories found
        dirs = re.findall(r'(https?://[^\s]+)\s+\(Status:\s+(\d+)', output)
        if dirs:
            summary += "Directories found:\n"
            for dir in dirs[:10]:  # Limit to first 10 for brevity
                summary += f"- {dir[0]} (Status: {dir[1]})\n"
            if len(dirs) > 10:
                summary += f"... and {len(dirs)-10} more\n"
    
    return summary

def is_command_safe(cmd):
    """
    Check if a command is safe to execute.
    This function performs additional security checks beyond the basic validation.
    
    Args:
        cmd (str): The command to check
        
    Returns:
        bool: True if the command is safe to execute
    """
    # List of forbidden commands that should never be run
    forbidden_commands = [
        'rm -rf /', 'rm -rf /*', 'mkfs', 'dd if=/dev/zero',
        'iptables -F', 'shutdown', 'reboot', 'halt',
        ':(){:|:&};:', 'chmod -R 777 /', 'chmod -R 000 /'
    ]
    
    # Check for forbidden commands
    for forbidden in forbidden_commands:
        if forbidden in cmd:
            return False
    
    # Check for commands that contain placeholders that need to be replaced
    placeholders = ['<target>', '<file>', '<domain>', '<ip>', '<port>', '<user>', '<password>']
    cmd_lower = cmd.lower()
    if any(placeholder in cmd_lower for placeholder in placeholders):
        return False
    
    # Check if this is an example command
    if 'example' in cmd_lower or '--help' in cmd_lower:
        # Allow help commands to run
        if '--help' in cmd and not any(keyword in cmd_lower for keyword in ['rm', 'delete', 'format']):
            return True
        return False
    
    # Special check for security tools
    security_keywords = ['nmap', 'nikto', 'metasploit', 'msfconsole', 'sqlmap', 
                         'hydra', 'hashcat', 'john', 'wireshark', 'tcpdump']
    
    if any(keyword in cmd_lower for keyword in security_keywords):
        # Allow scanning only specific IPs/domains
        if 'nmap' in cmd_lower:
            # Block scanning of common private networks unless explicitly allowed
            if any(net in cmd_lower for net in ['10.0.0.0', '172.16.0.0', '192.168.0.0']):
                # Block unless the scan is explicitly allowed
                if not 'ALLOWED_INTERNAL_SCAN' in cmd:
                    return False
        
        # Check for potential brute force attacks with large wordlists
        if ('hydra' in cmd_lower or 'hashcat' in cmd_lower or 'john' in cmd_lower) and 'rockyou' in cmd_lower:
            return False
    
    return True

def validate_api_endpoint(url):
    """
    Validate an API endpoint by testing its response.
    
    Args:
        url (str): The URL to test
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Test API endpoint
        test_result = subprocess.run(
            f'curl -sI {url}',
            shell=True, text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5  # 5 second timeout
        )
        if test_result.returncode != 0:
            return False, f"API endpoint test failed: {test_result.stderr}"
        
        # Get content type
        content_type = None
        for line in test_result.stdout.splitlines():
            if line.lower().startswith('content-type:'):
                content_type = line.split(':', 1)[1].strip()
                break
        
        # Check if it's JSON
        if content_type and 'json' in content_type.lower():
            json_test = subprocess.run(
                f'curl -s {url}',
                shell=True, text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if json_test.returncode != 0:
                return False, f"API request failed: {json_test.stderr}"
            try:
                json.loads(json_test.stdout)
            except json.JSONDecodeError:
                return False, "API response is not valid JSON"
        
        return True, None
        
    except subprocess.TimeoutExpired:
        return False, "API endpoint test timed out"
    except Exception as e:
        return False, f"API test error: {str(e)}"

def validate_command(cmd):
    """Validate and sanitize command input"""
    if not cmd:
        return None, "Empty command"
    
    cmd = cmd.strip('"\'')
    
    # Check for unsafe characters but allow piping for security tools
    unsafe_chars = [';', '&&', '||']
    security_keywords = ['nmap', 'nikto', 'metasploit', 'msfconsole', 'exploit', 'vuln', 
                         'sqlmap', 'burp', 'hydra', 'hashcat', 'john', 'wireshark', 'tcpdump']
    
    # Only check for unsafe characters if not a security command
    is_security_cmd = any(keyword in cmd.lower() for keyword in security_keywords)
    if not is_security_cmd and any(char in cmd for char in unsafe_chars):
        return None, "Invalid command: contains unsafe characters"
    
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return None, f"Invalid command format: {str(e)}"
    
    if not parts:
        return None, "Empty command after parsing"
    
    # Skip validation for security tools to allow more flexibility
    if is_security_cmd:
        return cmd, None
    
    # Check for incomplete flags
    if any(part.endswith('-') for part in parts):
        return None, "Incomplete command: incomplete flag detected"
    
    # Define command-specific validation
    command_validation = {
        'nmap': {
            'required_args': ['target'],
            'pattern': r'nmap\s+(?:-[a-zA-Z0-9]+\s+)*(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?|(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}',
            'error': "Nmap requires a valid IP address or hostname as a target"
        },
        'hashcat': {
            'required_args': ['hash_file', 'wordlist'],
            'pattern': r'hashcat\s+(?:-[a-zA-Z0-9]+\s+)*(\S+)\s+(\S+)',
            'error': "Hashcat requires a hash file and wordlist"
        },
        'gobuster': {
            'required_args': ['mode', 'url'],
            'pattern': r'gobuster\s+(?:dir|dns|vhost)\s+(?:-[a-zA-Z0-9]+\s+)*-u\s+\S+',
            'error': "Gobuster requires a mode (dir/dns/vhost) and a target URL (-u)"
        },
        'wireshark': {
            'required_args': ['interface'],
            'pattern': r'wireshark\s+(?:-[a-zA-Z0-9]+\s+)*-i\s+\S+',
            'error': "Wireshark requires an interface specification (-i)"
        },
        'sqlmap': {
            'required_args': ['url'],
            'pattern': r'sqlmap\s+(?:-[a-zA-Z0-9]+\s+)*-u\s+\S+',
            'error': "Sqlmap requires a target URL (-u)"
        }
    }
    
    # Get the base command
    base_cmd = parts[0] if parts else ""
    
    # Apply command-specific validation but only for non-security commands
    # or if this is a security command with validation pattern defined
    if base_cmd in command_validation and (not is_security_cmd or base_cmd in ['nmap', 'hashcat', 'gobuster', 'wireshark', 'sqlmap']):
        # Make validation more flexible for security tools
        if is_security_cmd:
            # Skip pattern validation for security tools
            pass
        else:
            validation = command_validation[base_cmd]
            if not re.match(validation['pattern'], cmd):
                return None, validation['error']
    
    # Handle placeholder values that need to be replaced
    placeholder_patterns = [
        r'<[^>]+>',              # <placeholder>
        r'\$\{[^}]+\}',          # ${placeholder}
        r'\[\s*[a-zA-Z_]+\s*\]', # [placeholder]
        r'\{[a-zA-Z_]+\}'        # {placeholder}
    ]
    
    # Skip placeholder check for security tools
    if not is_security_cmd:
        for pattern in placeholder_patterns:
            if re.search(pattern, cmd):
                return None, f"Command contains placeholder values that need to be replaced: {re.search(pattern, cmd).group(0)}"
    
    # Check if the command is a typical security tool without required flags - but make it more flexible
    security_tools_checks = {
        'nmap': ['-p', '--ports', '-sS', '-sV', '-A'],
        'sqlmap': ['-u', '--url', '-r', '--request'],
        'gobuster': ['-u', '--url'],
        'hashcat': ['-m', '--hash-type']
    }
    
    # Skip flag checks for security tools
    if base_cmd in security_tools_checks and not is_security_cmd:
        required_flags = security_tools_checks[base_cmd]
        if not any(flag in cmd for flag in required_flags):
            flag_list = ', '.join(required_flags)
            return None, f"{base_cmd} typically requires one of these flags: {flag_list}"
    
    # Check for example commands that shouldn't actually be run
    if not is_security_cmd and re.search(r'example|demo|syntax|placeholder|usage', cmd, re.IGNORECASE):
        return None, "This appears to be an example command and not meant for actual execution"
    
    # Special handling for curl commands with jq
    if cmd.startswith('curl') and '| jq' in cmd:
        url_match = re.search(r'https?://[^\s"]+', cmd)
        if url_match:
            url = url_match.group(0)
            is_valid, error = validate_api_endpoint(url)
            if not is_valid:
                return None, f"API validation failed: {error}"
            
            # Extract jq filter
            jq_match = re.search(r'jq\s+["\'](.+?)["\']', cmd)
            if jq_match:
                jq_filter = jq_match.group(1)
                # Test jq filter
                test_cmd = f'curl -s {url} | jq "{jq_filter}"'
                try:
                    test_result = subprocess.run(
                        test_cmd,
                        shell=True, text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    if test_result.returncode != 0:
                        return None, f"jq filter test failed: {test_result.stderr}"
                except Exception as e:
                    return None, f"jq filter test error: {str(e)}"
    
    return cmd, None

if __name__ == "__main__":
    # Simple self-test
    print("Command Handler Module Self-Test")
    
    test_cmd = "echo 'Hello, Nikita!'"
    hardened = harden_command(test_cmd)
    print(f"Original: {test_cmd}")
    print(f"Hardened: {hardened}")
    
    success, output = run_command(test_cmd)
    print(f"Success: {success}")
    print(f"Output: {output}") 