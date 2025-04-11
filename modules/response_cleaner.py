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
        
    def clean_response(self, response):
        """Clean and format the response"""
        # Remove internal reasoning markers
        response = re.sub(r'--- As Nikita, ---\n', '', response)
        response = re.sub(r'1\.\s+Understand the query:.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'1\.\s+Provide information:.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'1\.\s+Ask clarifying questions:.*?\n', '', response, flags=re.DOTALL)
        
        # Remove numbered steps that are part of internal reasoning
        response = re.sub(r'\d+\.\s+Understand.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'\d+\.\s+Provide.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'\d+\.\s+Ask.*?\n', '', response, flags=re.DOTALL)
        
        # Remove any remaining internal reasoning markers
        response = re.sub(r'Reasoning:.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'Analysis:.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'Thought process:.*?\n', '', response, flags=re.DOTALL)
        
        # Clean up any resulting double newlines
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        return response

    def format_for_display(self, response):
        """Format the response for display"""
        # Ensure response starts with a clean line
        response = response.lstrip()
        
        # Add proper spacing between sections
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        # Remove any remaining internal markers
        response = re.sub(r'\[Internal\]\s*', '', response)
        response = re.sub(r'\[Reasoning\]\s*', '', response)
        
        # Ensure response ends with a single newline
        response = response.rstrip() + '\n'
        
        return response


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
