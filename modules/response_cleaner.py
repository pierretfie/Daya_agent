#!/usr/bin/env python3
"""
Response Cleaner Module for Nikita Agent

Cleans and formats LLM responses to ensure they are human-readable
and free from metadata, JSON artifacts, and other unwanted content.
"""

import re
import json
from typing import Dict, Any, List, Union, Optional

class ResponseCleaner:
    """
    Cleans and formats raw LLM responses to ensure they are human-readable
    and consistent with Nikita's expected output format.
    """
    
    def __init__(self):
        """Initialize the response cleaner"""
        self.json_pattern = re.compile(r'^\s*\{.*\}\s*$', re.DOTALL)
        self.command_pattern = re.compile(r'```(?:\w+)?\s*([^`]+)```|`([^`]+)`')
        self.title_only_pattern = re.compile(r'^---\s*(What|How|Why|When|Where|Who)\s+.*\s*---$', re.IGNORECASE)
        
        # Add pattern to detect questions in user input
        self.question_pattern = re.compile(r'^(what|how|why|when|where|who)\s+(is|are|does|do|can|should|would)\s+', re.IGNORECASE)
        
        # Role prefixes to clean from responses
        self.role_prefixes = [
            r'^\s*Nikita\s*:\s*',
            r'^\s*NIKITA\s*:\s*',
            r'^\s*Assistant\s*:\s*',
            r'^\s*ASSISTANT\s*:\s*',
            r'^\s*User\s*:\s*',
            r'^\s*USER\s*:\s*',
            r'^\s*Human\s*:\s*',
            r'^\s*HUMAN\s*:\s*',
            r'^\s*AI\s*:\s*',
            r'^\s*System\s*:\s*',
            r'^\s*SYSTEM\s*:\s*'
        ]
        
        # Compile role prefix patterns for efficiency
        self.role_prefix_patterns = [re.compile(pattern) for pattern in self.role_prefixes]
        
        # Metadata sections to extract
        self.metadata_sections = [
            "response_strategy", "execution_plan", "context", "reasoning",
            "technical_context", "emotional_context", "personal_context",
            "domain", "intent", "answered_context", "follow_up_questions"
        ]
        
        # Security-related action patterns
        self.security_action_patterns = {
            "exploit": "I understand you're interested in exploitation. Let me explain the security implications and ethical considerations. What specific aspect would you like to learn about?",
            "hack": "I can help explain security concepts and defensive measures. What specific security topic would you like to understand better?",
            "attack": "I can help you understand security concepts and defense strategies. What specific aspect of security would you like to learn about?",
            "scan": "I can explain network scanning concepts, tools, and best practices. Would you like to learn about security assessment methodologies?",
            "gather": "I can explain information gathering techniques and their security implications. What specific aspect interests you?",
            "reconnaissance": "I can help you understand reconnaissance and its role in security assessment. What would you like to know?"
        }
        
        # New patterns for removing reasoning steps and analysis indicators
        self.reasoning_patterns = [
            # Pattern for numbered steps with labels (e.g., "1. UNDERSTAND: ...")
            re.compile(r'^\s*\d+\.\s*[A-Z_]+:.*$', re.MULTILINE),
            
            # Pattern for numbered steps (e.g., "1. Understand the query: ...")
            re.compile(r'^\s*\d+\.\s*(Understand|Analyze|Identify|Determine|Provide|Ask).*:.*$', re.MULTILINE),
            
            # Pattern for "As Nikita, ..." prefix
            re.compile(r'^\s*---\s*As Nikita,\s*---\s*$', re.MULTILINE),
            re.compile(r'^\s*As Nikita,\s*', re.MULTILINE),
            
            # Pattern for task analysis headings
            re.compile(r'^\s*Task Analysis:.*$', re.MULTILINE),
            
            # Pattern for response structure indicators
            re.compile(r'^\s*(Understanding|Analysis|Approach|Response):.*$', re.MULTILINE)
        ]
        
    def clean_response(self, response):
        """Clean up the response text and extract components"""
        if not response or response.strip() == "":
            return {
                "text": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question.",
                "commands": [],
                "codeblocks": [],
                "placeholders": {}
            }
            
        # Check for extremely short responses (likely issues with response generation)
        if len(response.strip()) < 20:
            # Handle security-related short responses
            response_lower = response.strip().lower()
            
            # Check for security action patterns
            for action, explanation in self.security_action_patterns.items():
                if action in response_lower or f"--- {action}" in response_lower:
                    return {
                        "text": explanation,
                        "commands": [],
                        "codeblocks": [],
                        "placeholders": {}
                    }
            
            # Handle generic short responses
            if response.strip() in ["---", "--- Perform ---", "--- Exploiting ---", "--- Information Gathering ---"]:
                return {
                    "text": "I understand you're asking about security-related actions. I can help explain security concepts, methodologies, and best practices. Could you provide more details about what specific information you're looking for?",
                    "commands": [],
                    "codeblocks": [],
                    "placeholders": {}
                }
                
        # Check for title-only responses (e.g., "--- What Is Burpsuite ---")
        title_only_pattern = re.compile(r'^---\s*(What|How|Why|When|Where|Who)\s+.*\s*---$', re.IGNORECASE)
        if title_only_pattern.match(response.strip()):
            # Extract the topic from the title
            topic_match = re.search(r'---\s*(What|How|Why|When|Where|Who)\s+(.*?)\s*---', response.strip(), re.IGNORECASE)
            if topic_match:
                question_type = topic_match.group(1).lower()
                topic = topic_match.group(2).strip()
                
                # Handle cases like "Is X" where X is the actual topic
                if topic.lower().startswith("is "):
                    topic = topic[3:].strip()
                elif topic.lower().startswith("are "):
                    topic = topic[4:].strip()
                elif topic.lower().startswith("does "):
                    topic = topic[5:].strip()
                elif topic.lower().startswith("do "):
                    topic = topic[3:].strip()
                
                # Try to get a more complete response by using the single word topic
                # This helps with cases where "what is X" fails but just "X" works
                if len(topic.split()) >= 1:
                    # Get the main topic word (usually the first word or two)
                    main_topic = topic.split()[0].lower()
                    
                    # For compound topics like "burp suite", include both words
                    if len(topic.split()) > 1 and len(topic.split()[0]) <= 5:
                        main_topic = " ".join(topic.split()[:2]).lower()
                
                # Generate a helpful fallback response based on the question type and topic
                if question_type == 'what':
                    return {
                        "text": f"I'd be happy to explain about {topic}. To get a complete response, try asking just '{topic}' without the 'what is' prefix, or ask about a specific aspect of {topic} you're interested in.",
                        "commands": [],
                        "codeblocks": [],
                        "placeholders": {}
                    }
                elif question_type == 'how':
                    return {
                        "text": f"I'd be happy to explain how to use or work with {topic}. To get a complete response, try asking just '{topic}' without the 'how' prefix, or ask about a specific aspect of {topic} you're interested in.",
                        "commands": [],
                        "codeblocks": [],
                        "placeholders": {}
                    }
                else:
                    return {
                        "text": f"I'd be happy to provide information about {topic}. To get a complete response, try asking just '{topic}' without the question prefix, or ask about a specific aspect of {topic} you're interested in.",
                        "commands": [],
                        "codeblocks": [],
                        "placeholders": {}
                    }
                
        # Check if the response contains command markers
        command_result = self._extract_commands(response)
        
        # Extract code blocks
        code_result = self._extract_code_blocks(command_result["text"])
        
        # Normalize the response format
        normalized = self._normalize_text(code_result["text"])
        
        # Replace placeholders with their values
        cleaned_text = self._replace_placeholders(normalized["text"], normalized["placeholders"])
        
        # Check if the cleaned text is too short or generic
        if len(cleaned_text.strip()) < 50:  # Increased minimum length threshold
            # Look for security-related content
            text_lower = cleaned_text.lower()
            for action, explanation in self.security_action_patterns.items():
                if action in text_lower:
                    cleaned_text = explanation
                    break
                    
            # Check for title-only responses that might have passed earlier checks
            title_only_pattern = re.compile(r'^---\s*(What|How|Why|When|Where|Who)\s+.*\s*---$', re.IGNORECASE)
            if title_only_pattern.match(cleaned_text.strip()):
                # Extract the topic from the title
                topic_match = re.search(r'---\s*(What|How|Why|When|Where|Who)\s+(.*?)\s*---', cleaned_text.strip(), re.IGNORECASE)
                if topic_match:
                    question_type = topic_match.group(1).lower()
                    topic = topic_match.group(2).strip()
                    
                    # Handle cases like "Is X" where X is the actual topic
                    if topic.lower().startswith("is "):
                        topic = topic[3:].strip()
                    
                    # Generate a helpful fallback response
                    cleaned_text = f"I'd be happy to provide information about {topic}. To get a complete response, try asking just '{topic}' without the question prefix, or ask about a specific aspect of {topic} you're interested in."
        
        # Apply final formatting
        result = {
            "text": cleaned_text,
            "commands": command_result["commands"],
            "codeblocks": code_result["codeblocks"],
            "placeholders": normalized["placeholders"]
        }
        
        return result
        
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text if present"""
        if self.json_pattern.match(text):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
                
        # Try to find JSON within text
        try:
            # Look for JSON-like structures
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
            
        return None
        
    def _process_json_response(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON-formatted response"""
        clean_text = ""
        metadata = {}
        commands = []
        
        # Extract the main response text
        if 'response' in json_data and isinstance(json_data['response'], dict):
            if 'text' in json_data['response']:
                clean_text = json_data['response']['text']
            
            # Move context to metadata
            if 'context' in json_data['response']:
                metadata['context'] = json_data['response']['context']
                
        elif 'text' in json_data:
            clean_text = json_data['text']
        elif 'output' in json_data:
            clean_text = json_data['output']
        elif 'content' in json_data:
            clean_text = json_data['content']
        elif 'message' in json_data:
            clean_text = json_data['message']
        elif 'answer' in json_data:
            clean_text = json_data['answer']
        
        # If we still don't have clean text, use the first string value we find
        if not clean_text:
            for key, value in json_data.items():
                if isinstance(value, str) and len(value) > 20:
                    clean_text = value
                    break
        
        # If still no clean text, use the entire JSON as a fallback
        if not clean_text:
            clean_text = "I apologize, but I couldn't format my response properly. Here's the raw data:\n\n" + json.dumps(json_data, indent=2)
            
        # Remove any role prefixes from the beginning of the response
        clean_text = self._remove_role_prefixes(clean_text)
            
        # Extract metadata
        for section in self.metadata_sections:
            if section in json_data:
                metadata[section] = json_data[section]
                
        # Extract commands from the clean text
        commands = self._extract_commands(clean_text)
        
        return {
            'clean_text': clean_text,
            'commands': commands,
            'metadata': metadata
        }
        
    def _process_text_response(self, text: str) -> Dict[str, Any]:
        """Process a plain text response"""
        # Remove any role prefixes from the beginning of the response
        text = self._remove_role_prefixes(text)
        
        # Remove reasoning steps and analysis indicators
        text = self._remove_reasoning_patterns(text)
        
        # Split into lines for processing
        lines = text.split('\n')
        clean_lines = []
        metadata = {}
        skip_section = False
        current_section = None
        section_content = []
        
        # Process line by line
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            section_match = re.match(r'^#+\s*(.*?)\s*:?\s*$', line)
            if section_match:
                section_name = section_match.group(1).lower()
                
                # If ending a metadata section, store it
                if current_section in self.metadata_sections:
                    metadata[current_section] = '\n'.join(section_content)
                    section_content = []
                
                # Check if this is a metadata section
                if any(meta in section_name for meta in self.metadata_sections):
                    current_section = section_name
                    skip_section = True
                    continue
                else:
                    current_section = None
                    skip_section = False
            
            # If in a metadata section, collect content
            if skip_section or current_section in self.metadata_sections:
                section_content.append(line)
                continue
                
            # Skip JSON-like lines
            if line.startswith('{') and line.endswith('}'):
                continue
                
            # Skip empty lines at the beginning
            if not clean_lines and not line:
                continue
                
            # Add the line to clean lines
            clean_lines.append(line)
            
        # Store the last section if it was metadata
        if current_section in self.metadata_sections:
            metadata[current_section] = '\n'.join(section_content)
            
        # Join clean lines back together
        clean_text = '\n'.join(clean_lines)
        
        # Extract commands
        commands = self._extract_commands(clean_text)
        
        return {
            'clean_text': clean_text,
            'commands': commands,
            'metadata': metadata
        }
        
    def _extract_commands(self, text: str) -> Dict[str, Any]:
        """Extract commands from code blocks in text"""
        commands = []
        
        # Find all code blocks
        matches = self.command_pattern.findall(text)
        for match in matches:
            # Each match is a tuple with groups from the regex
            command = match[0] if match[0] else match[1]
            if command:
                commands.append(command.strip())
        
        # Return both commands and text
        return {
            "text": text,
            "commands": commands
        }
        
    def _extract_code_blocks(self, text: str) -> Dict[str, Any]:
        """Extract code blocks from text"""
        code_blocks = []
        
        # Pattern to match code blocks with language specifiers
        code_pattern = re.compile(r'```(?:\w+)?\s*([^`]+)```', re.MULTILINE | re.DOTALL)
        
        # Find and extract all code blocks
        matches = code_pattern.findall(text)
        for idx, match in enumerate(matches):
            code_blocks.append({
                "content": match.strip(),
                "language": "unknown",  # Default language
                "id": f"code_{idx}"
            })
            
        # Replace code blocks with placeholders
        cleaned_text = text
        for idx, block in enumerate(code_blocks):
            placeholder = f"[CODE_BLOCK_{idx}]"
            cleaned_text = code_pattern.sub(placeholder, cleaned_text, count=1)
            
        return {
            "text": cleaned_text,
            "codeblocks": code_blocks
        }
        
    def _normalize_text(self, text: str) -> Dict[str, Any]:
        """Normalize text by removing special formatting and replacing with placeholders"""
        placeholders = {}
        
        # Remove any role prefixes
        text = self._remove_role_prefixes(text)
        
        # Remove reasoning patterns
        text = self._remove_reasoning_patterns(text)
        
        # Replace section headers with cleaner format
        header_pattern = re.compile(r'^#{1,6}\s*(.*?)\s*$', re.MULTILINE)
        headers = header_pattern.findall(text)
        
        for idx, header in enumerate(headers):
            placeholder = f"[HEADER_{idx}]"
            placeholders[placeholder] = f"**{header}**"
            text = header_pattern.sub(placeholder, text, count=1)
            
        # Normalize bullet points
        bullet_pattern = re.compile(r'^\s*[-*•]\s+(.*?)$', re.MULTILINE)
        bullets = bullet_pattern.findall(text)
        
        for idx, bullet in enumerate(bullets):
            placeholder = f"[BULLET_{idx}]"
            placeholders[placeholder] = f"• {bullet}"
            text = bullet_pattern.sub(placeholder, text, count=1)
            
        # Normalize line breaks (replace 3+ consecutive newlines with 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return {
            "text": text,
            "placeholders": placeholders
        }
        
    def _replace_placeholders(self, text: str, placeholders: Dict[str, str]) -> str:
        """Replace placeholders with their values"""
        result = text
        
        # Replace each placeholder with its value
        for placeholder, value in placeholders.items():
            result = result.replace(placeholder, value)
            
        return result
        
    def _remove_role_prefixes(self, text: str) -> str:
        """Remove role prefixes from text"""
        if not text:
            return text
            
        # Apply all role prefix patterns
        for pattern in self.role_prefix_patterns:
            text = pattern.sub('', text)
            
        # Also handle multi-line responses with role prefixes
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line
            for pattern in self.role_prefix_patterns:
                cleaned_line = pattern.sub('', cleaned_line)
            cleaned_lines.append(cleaned_line)
            
        return '\n'.join(cleaned_lines)
        
    def _remove_reasoning_patterns(self, text: str) -> str:
        """Remove reasoning steps and analysis indicators from text"""
        if not text:
            return text
            
        # Apply all reasoning patterns
        for pattern in self.reasoning_patterns:
            # Replace matching patterns with empty lines
            text = pattern.sub('', text)
        
        # Clean up multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up leading and trailing whitespace
        text = text.strip()
        
        return text
        
    def format_for_display(self, cleaned_result: Dict[str, Any]) -> str:
        """Format the cleaned result for display to the user"""
        if not cleaned_result:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        if "text" in cleaned_result:
            return cleaned_result["text"]
        elif "clean_text" in cleaned_result:
            return cleaned_result["clean_text"]
        else:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."


if __name__ == "__main__":
    # Simple self-test
    cleaner = ResponseCleaner()
    
    # Test with JSON response
    json_response = '''
    {
      "response_strategy": {
        "approach": "informative",
        "tone": "helpful",
        "technical_level": "moderate",
        "follow_up_questions": []
      },
      "execution_plan": {
        "steps": [
          "understand query",
          "provide information",
          "ask clarifying questions"
        ],
        "priority": "normal",
        "dependencies": [],
        "command": null
      },
      "response": {
        "text": "To attack a target like google.com, you need to understand the scope of your operations and obtain explicit authorization. Once you have that in place, you can start gathering information about the target.",
        "context": {
          "domain": "general",
          "intent": "information_request",
          "personal_context": null,
          "technical_context": {
            "task_type": "targeted attack",
            "target": "google.com"
          },
          "emotional_context": null,
          "answered_context": null
        }
      }
    }
    '''
    
    cleaned = cleaner.clean_response(json_response)
    print("Cleaned JSON Response:")
    print(cleaner.format_for_display(cleaned))
    print("\nExtracted Commands:", cleaned['commands'])
    print("\nMetadata:", cleaned['metadata'])
    
    # Test with text response
    text_response = '''
    # Response Strategy
    This is a helpful response
    
    To use nmap for scanning, you can run:
    ```
    nmap -sS -p 1-1000 192.168.1.1
    ```
    
    Or for a simple ping:
    `ping -c 4 google.com`
    
    # Technical Context
    This is some technical context
    '''
    
    cleaned = cleaner.clean_response(text_response)
    print("\n\nCleaned Text Response:")
    print(cleaner.format_for_display(cleaned))
    print("\nExtracted Commands:", cleaned['commands'])
    print("\nMetadata:", cleaned['metadata'])
