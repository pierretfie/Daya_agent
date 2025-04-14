#!/usr/bin/env python3
"""
Tool Manager Module for Daya Agent

Handles tool-related functionality including man pages, help information,
and tool context management.
"""

import subprocess
import re
import json
from pathlib import Path
from rich.console import Console
import shlex
from .documentation_verifier import DocumentationVerifier

console = Console()

class ToolManager:
    def __init__(self, fine_tuning_file=None):
        """
        Initialize the tool manager.
        
        Args:
            fine_tuning_file (str, optional): Path to fine-tuning data file
        """
        self.fine_tuning_file = fine_tuning_file
        self.tool_cache = {}
        self.documentation_verifier = DocumentationVerifier()
        
        # Base template for all security tools
        self.tool_template = {
            "purpose": "",
            "category": "",
            "legitimate_uses": [],
            "syntax": "",
            "parameters": {},
            "examples": {},
            "ethical_notice": "This tool should only be used for legitimate security testing with proper authorization."
        }

    def get_tool_manpage(self, tool_name):
        """Fetch and parse man page for a security tool"""
        try:
            # Run man command and capture output
            result = subprocess.run(['man', tool_name], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch man page for {tool_name}: {str(e)}[/yellow]")
            return None

    def parse_manpage(self, manpage_content):
        """Parse man page content to extract useful information"""
        if not manpage_content:
            return None
        
        # Extract common sections
        sections = {
            "name": re.search(r"NAME\n\s*(.*?)\n", manpage_content),
            "synopsis": re.search(r"SYNOPSIS\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "description": re.search(r"DESCRIPTION\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "options": re.search(r"OPTIONS\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL),
            "examples": re.search(r"EXAMPLES\n(.*?)\n(?=\w|$)", manpage_content, re.DOTALL)
        }
        
        parsed = {}
        for section, match in sections.items():
            if match:
                parsed[section] = match.group(1).strip()
        
        return parsed

    def get_tool_help(self, tool_name):
        """Get help information for a security tool with improved summarization"""
        # Check cache first
        if tool_name in self.tool_cache:
            return self.tool_cache[tool_name]
        
        # First try to get man page
        manpage = self.get_tool_manpage(tool_name)
        if manpage:
            parsed = self.parse_manpage(manpage)
            if parsed:
                # Generate summarized content
                summary = self._summarize_manpage_content(parsed)
                
                # Verify documentation
                is_verified, verification_details = self.documentation_verifier.verify_tool_documentation(
                    tool_name, 
                    summary
                )
                
                # Format the help information
                help_info = {
                    "source": "man_page",
                    "name": tool_name,
                    "formatted_help": self.format_tool_help(tool_name, summary),
                    "raw_summary": summary,  # Keep the raw summary for potential other uses
                    "verification": {
                        "is_verified": is_verified,
                        "details": verification_details
                    }
                }
                
                # Update local knowledge base with verified documentation
                if is_verified:
                    self.documentation_verifier.update_local_knowledge_base(
                        tool_name,
                        summary,
                        verification_details
                    )
                
                self.tool_cache[tool_name] = help_info
                return help_info
        
        # Fallback to --help if man page not available
        try:
            result = subprocess.run([tool_name, '--help'], capture_output=True, text=True)
            if result.returncode == 0:
                help_info = {
                    "source": "help_flag",
                    "help_text": result.stdout
                }
                self.tool_cache[tool_name] = help_info
                return help_info
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get help for {tool_name}: {str(e)}[/yellow]")
        
        return None

    def get_tool_context(self, tool_name):
        """Get comprehensive context for a security tool"""
        context = {
            "man_page": None,
            "fine_tuning": None,
            "common_usage": None,
            "verified_documentation": None,
            "citations": None
        }
        
        # Get man page information
        tool_help = self.get_tool_help(tool_name)
        if tool_help:
            context["man_page"] = tool_help
        
        # Get fine-tuning data if file is available
        if self.fine_tuning_file and Path(self.fine_tuning_file).exists():
            try:
                with open(self.fine_tuning_file, "r") as f:
                    fine_tuning_data = json.load(f)
                    tool_data = [entry for entry in fine_tuning_data if entry.get("tool_used") == tool_name]
                    if tool_data:
                        context["fine_tuning"] = tool_data
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load fine-tuning data for {tool_name}: {str(e)}[/yellow]")
        
        # Get common usage patterns
        if tool_name in self.common_usage:
            context["common_usage"] = self.common_usage[tool_name]
        
        # Get verified documentation from local knowledge base
        verified_doc = self.documentation_verifier.get_local_documentation(tool_name)
        if verified_doc:
            context["verified_documentation"] = verified_doc
        
        # Get citations
        citations = self.documentation_verifier.get_citations(tool_name)
        if citations:
            context["citations"] = citations
        
        return context

    def clear_cache(self):
        """Clear the tool information cache"""
        self.tool_cache.clear()

    def get_security_tool_info(self, tool_name):
        """Get comprehensive, responsible information about security tools"""
        # Check cache first
        if tool_name in self.tool_cache:
            return self.tool_cache[tool_name]

        tool_info = self.tool_template.copy()

        # Try multiple sources to gather tool information
        sources = [
            self._get_man_page_info,    # Get info from man pages
            self._get_help_info,        # Get info from --help
            self._get_package_info,     # Get info from package metadata
            self._get_online_docs       # Get info from local documentation
        ]

        for source in sources:
            info = source(tool_name)
            if info:
                self._merge_tool_info(tool_info, info)

        # Add ethical guidelines
        tool_info["ethical_notice"] = self._generate_ethical_notice(tool_info["category"])
        
        # Cache the result
        self.tool_cache[tool_name] = tool_info
        return tool_info

    def _get_man_page_info(self, tool_name):
        """Extract structured information from man pages"""
        man_info = self.get_tool_manpage(tool_name)
        if not man_info:
            return None

        parsed = self.parse_manpage(man_info)
        if not parsed:
            return None

        # Convert man page sections to our format
        return {
            "purpose": parsed.get("description", "").split(".")[0],
            "syntax": parsed.get("synopsis", ""),
            "parameters": self._extract_parameters(parsed.get("options", "")),
            "examples": self._extract_examples(parsed.get("examples", ""))
        }

    def _get_help_info(self, tool_name):
        """Extract information from --help output"""
        try:
            result = subprocess.run([tool_name, '--help'], 
                                 capture_output=True, 
                                 text=True,
                                 timeout=5)
            
            if result.returncode == 0:
                return {
                    "syntax": self._extract_syntax(result.stdout),
                    "parameters": self._extract_parameters(result.stdout),
                    "examples": self._extract_examples(result.stdout)
                }
        except:
            return None

    def _extract_parameters(self, text):
        """Extract parameters and their descriptions from text"""
        params = {}
        # Use regex to find parameter patterns like -p, --param
        param_pattern = r'-(\w),?\s+--?([\w-]+)\s+(.+?)(?=\n\s*-|\Z)'
        matches = re.finditer(param_pattern, text, re.MULTILINE)
        
        for match in matches:
            short_param = f"-{match.group(1)}"
            long_param = f"--{match.group(2)}"
            description = match.group(3).strip()
            params[short_param] = description
            params[long_param] = description

        return params

    def _extract_examples(self, text):
        """Extract example commands from text"""
        examples = {}
        # Look for common example patterns
        example_pattern = r'(?:Example|e\.g\.)[:\s]+([^\n]+)'
        matches = re.finditer(example_pattern, text, re.IGNORECASE)
        
        for i, match in enumerate(matches, 1):
            examples[f"example_{i}"] = match.group(1).strip()

        return examples

    def _generate_ethical_notice(self, category):
        """Generate appropriate ethical notice based on tool category"""
        notices = {
            "scanner": "This scanning tool should only be used on systems you own or have explicit permission to test.",
            "exploit": "This exploitation tool should only be used in authorized penetration testing environments.",
            "crypto": "This cryptographic tool should be used responsibly and in compliance with applicable laws.",
            "forensic": "This forensic tool should be used within appropriate legal and ethical boundaries.",
            "default": "This security tool should only be used for legitimate security testing with proper authorization."
        }
        return notices.get(category, notices["default"])

    def _merge_tool_info(self, base_info, new_info):
        """Merge new tool information into base info"""
        for key, value in new_info.items():
            if key in base_info:
                if isinstance(base_info[key], dict):
                    base_info[key].update(value)
                elif isinstance(base_info[key], list):
                    base_info[key].extend(value)
                else:
                    base_info[key] = value

    def _summarize_manpage_content(self, parsed_content):
        """Summarize and paraphrase man page content into clear, digestible sections"""
        summary = {
            "quick_overview": "",
            "key_features": [],
            "common_usage": [],
            "important_flags": {},
            "security_notes": [],
            "examples_explained": []
        }

        # Generate quick overview from description
        if description := parsed_content.get('description'):
            # Take first sentence or first 100 characters, whichever is shorter
            first_sentence = description.split('.')[0]
            summary["quick_overview"] = (
                first_sentence[:100] + '...' if len(first_sentence) > 100 
                else first_sentence
            )

        # Extract and summarize key features
        if description := parsed_content.get('description'):
            # Look for feature indicators
            feature_indicators = ['can', 'allows', 'supports', 'provides', 'enables']
            sentences = description.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in feature_indicators):
                    # Clean and simplify the feature description
                    feature = sentence.strip().replace('\n', ' ').split('  ')[0]
                    if len(feature) > 10:  # Avoid tiny fragments
                        summary["key_features"].append(feature)

        # Summarize common usage patterns
        if synopsis := parsed_content.get('synopsis'):
            # Extract basic usage patterns
            usage_patterns = synopsis.split('\n')
            for pattern in usage_patterns:
                if pattern.strip() and not pattern.startswith('or'):
                    # Simplify and clean up the pattern
                    cleaned = ' '.join(pattern.split())
                    if len(cleaned) > 10:
                        summary["common_usage"].append(cleaned)

        # Extract and explain important flags
        if options := parsed_content.get('options'):
            flag_pattern = r'-(\w),?\s+--?([\w-]+)(?:\s+\w+)?\s+(.+?)(?=\n\s*-|\Z)'
            matches = re.finditer(flag_pattern, options, re.MULTILINE)
            
            for match in matches:
                flag = f"-{match.group(1)}"
                description = match.group(3).strip()
                
                # Simplify long descriptions
                if len(description) > 100:
                    description = description.split('.')[0] + '.'
                
                summary["important_flags"][flag] = description

        # Extract and explain examples
        if examples := parsed_content.get('examples'):
            example_pattern = r'(?:Example|e\.g\.)[:\s]+([^\n]+)'
            matches = re.finditer(example_pattern, examples, re.IGNORECASE)
            
            for match in matches:
                example = match.group(1).strip()
                # Add a brief explanation
                explanation = self._generate_example_explanation(example)
                summary["examples_explained"].append({
                    "command": example,
                    "explanation": explanation
                })

        # Extract security-relevant information
        security_keywords = ['security', 'permission', 'privilege', 'risk', 'warning', 'caution']
        for section, content in parsed_content.items():
            if isinstance(content, str):
                for keyword in security_keywords:
                    if keyword in content.lower():
                        # Extract the relevant sentence or paragraph
                        context = self._extract_security_context(content, keyword)
                        if context:
                            summary["security_notes"].append(context)

        return summary

    def _generate_example_explanation(self, example):
        """Generate a human-friendly explanation of a command example"""
        parts = shlex.split(example)
        if not parts:
            return "Empty example"

        explanation = []
        command = parts[0]
        flags = [p for p in parts[1:] if p.startswith('-')]
        targets = [p for p in parts[1:] if not p.startswith('-')]

        # Explain the base command
        explanation.append(f"Uses {command}")

        # Explain flags
        if flags:
            explanation.append("with options:")
            for flag in flags:
                if flag in self.tool_cache.get(command, {}).get('parameters', {}):
                    explanation.append(f"- {flag}: {self.tool_cache[command]['parameters'][flag]}")
                else:
                    explanation.append(f"- {flag}")

        # Explain targets/arguments
        if targets:
            explanation.append("operating on:")
            for target in targets:
                explanation.append(f"- {target}")

        return " ".join(explanation)

    def _extract_security_context(self, content, keyword):
        """Extract relevant security context around a keyword"""
        # Find the sentence containing the keyword
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            if keyword in sentence.lower():
                # Clean up and simplify the sentence
                cleaned = sentence.strip().replace('\n', ' ').split('  ')[0]
                if len(cleaned) > 150:
                    # If too long, try to extract the most relevant part
                    parts = cleaned.split(',')
                    for part in parts:
                        if keyword in part.lower():
                            return part.strip()
                return cleaned
        return None

    def format_tool_help(self, tool_name, summary):
        """Format the summarized information into a clear, readable response"""
        sections = []
        
        # Add overview
        if summary["quick_overview"]:
            sections.append(f"Overview:\n{summary['quick_overview']}")

        # Add key features
        if summary["key_features"]:
            sections.append("Key Features:\n" + "\n".join(
                f"• {feature}" for feature in summary["key_features"][:5]
            ))

        # Add common usage
        if summary["common_usage"]:
            sections.append("Common Usage:\n" + "\n".join(
                f"• {usage}" for usage in summary["common_usage"][:3]
            ))

        # Add important flags
        if summary["important_flags"]:
            sections.append("Important Flags:\n" + "\n".join(
                f"• {flag}: {desc}" for flag, desc in 
                list(summary["important_flags"].items())[:5]
            ))

        # Add examples with explanations
        if summary["examples_explained"]:
            sections.append("Examples:\n" + "\n".join(
                f"• {ex['command']}\n  {ex['explanation']}" 
                for ex in summary["examples_explained"][:3]
            ))

        # Add security notes if any
        if summary["security_notes"]:
            sections.append("Security Notes:\n" + "\n".join(
                f"• {note}" for note in summary["security_notes"]
            ))

        return "\n\n".join(sections)

if __name__ == "__main__":
    # Simple self-test
    tool_manager = ToolManager()
    
    # Test with nmap
    nmap_context = tool_manager.get_tool_context("nmap")
    print("Nmap Context:")
    print(json.dumps(nmap_context, indent=2)) 