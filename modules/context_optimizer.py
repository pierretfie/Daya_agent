#!/usr/bin/env python3
"""
Context Optimizer Module for Nikita Agent

Optimizes conversation context for LLM interactions by selecting relevant
messages, handling token limits, and improving prompt quality.
"""

import re
import psutil
from datetime import datetime
import json

# Default token limits
DEFAULT_MAX_TOKENS = 15000 #increase from 2048 due to large input
DEFAULT_RESERVE_TOKENS = 512

class ContextOptimizer:
    def __init__(self, max_tokens=DEFAULT_MAX_TOKENS, reserve_tokens=DEFAULT_RESERVE_TOKENS):
        """
        Initialize the context optimizer.
        
        Args:
            max_tokens (int): Maximum number of tokens for response generation
            reserve_tokens (int): Number of tokens to reserve for response
        """
        self.max_tokens = max_tokens  # Maximum tokens for response generation
        self.reserve_tokens = reserve_tokens  # Tokens to reserve for response
        self.context_window = DEFAULT_MAX_TOKENS  # context window size
        self.prompt_cache = {}
        self.engagement_memory = None
        
    def format_tool_context(self, tool_context):
        """Format tool context into a readable string for the model"""
        if not tool_context:
            return ""
            
        formatted = []
        
        # Format man page information
        if tool_context.get("man_page"):
            man_page = tool_context["man_page"]
            formatted.append("Tool Documentation:")
            if man_page.get("name"):
                formatted.append(f"Name: {man_page['name']}")
            if man_page.get("synopsis"):
                formatted.append(f"Usage: {man_page['synopsis']}")
            if man_page.get("description"):
                formatted.append(f"Description: {man_page['description']}")
            if man_page.get("options"):
                formatted.append(f"Options: {man_page['options']}")
            if man_page.get("examples"):
                formatted.append(f"Examples: {man_page['examples']}")
        
        # Format fine-tuning data
        if tool_context.get("fine_tuning"):
            formatted.append("\nCommon Use Cases:")
            for entry in tool_context["fine_tuning"]:
                formatted.append(f"- {entry.get('instruction', '')}")
                if entry.get("command"):
                    formatted.append(f"  Command: {entry['command']}")
        
        # Format common usage patterns
        if tool_context.get("common_usage"):
            formatted.append("\nCommon Usage Patterns:")
            for pattern_name, pattern in tool_context["common_usage"].items():
                formatted.append(f"- {pattern_name}: {pattern}")
        
        return "\n".join(formatted)

    def optimize_context(self, chat_memory, current_task, targets=None):
        """
        Optimize context window by selecting relevant messages.
        
        Args:
            chat_memory (list): List of chat messages (dicts with 'role', 'content')
            current_task (str): Current user task/query
            targets (list, optional): List of targets (IPs, etc.) to prioritize
            
        Returns:
            list: List of relevant context messages
        """
        # Check cache first for performance
        cache_key = f"{current_task}_{len(chat_memory)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Handle empty history
        if not chat_memory:
            return []
            
        # Process only recent messages to save processing time
        recent_messages = chat_memory[-min(self.memory_limit, len(chat_memory)):]
        
        # Faster relevance scoring - avoid complex calculations
        scored_messages = []
        
        # Focus on just the last 15 messages for faster processing
        if len(recent_messages) <= 15:
            # Just return all messages if 15 or fewer
            relevant_msgs = [msg['content'] for msg in recent_messages if isinstance(msg, dict) and msg.get('content')]
            self.cache[cache_key] = relevant_msgs
            return relevant_msgs
            
        # Get last 15 messages directly - fast path optimization
        relevant_msgs = [msg['content'] for msg in recent_messages[-15:] 
                         if isinstance(msg, dict) and msg.get('content')]
        
        # Cache the result
        self.cache[cache_key] = relevant_msgs
        
        # Limit cache size to prevent memory growth
        if len(self.cache) > 50:
            # Remove oldest entries (simple approach)
            keys_to_remove = list(self.cache.keys())[:-25]  # Keep 25 newest items
            for key in keys_to_remove:
                self.cache.pop(key, None)
                
        return relevant_msgs

    def get_optimized_prompt(self, chat_memory, current_task, base_prompt, reasoning_context=None, 
                           follow_up_questions=None, tool_context=None):
        """
        Get an optimized prompt with context for the LLM.
        """
        # Enhanced base prompt for security-focused explanations
        if not base_prompt:
            base_prompt = """You are Nikita 🐺, an Offline AI Security Assistant specializing in clear, structured explanations of security tools and concepts. When responding:

FOR SECURITY TOOL EXPLANATIONS:
1. Start with a 1-2 sentence overview of what the tool is and its primary purpose
2. List key capabilities and features in bullet points (3-5 points)
3. Provide basic syntax using code blocks with clear formatting, showing main flags/arguments
4. Explain relevant security implications and potential attack/defense scenarios
5. Show 1-2 practical example commands with brief explanations of what they do
6. Include potential risks, ethical considerations, or legal implications if relevant

FOR SECURITY CONCEPT EXPLANATIONS:
1. Begin with a clear definition of the concept in 1-2 sentences
2. Explain the security relevance and why it matters
3. Describe how the concept is applied in real-world security contexts
4. Include related threats or vulnerabilities
5. Provide mitigation strategies or best practices
6. Reference related security tools or technologies if applicable

IMPORTANT GUIDELINES:
- Only suggest commands when explicitly requested with phrases like "run", "execute", or "show me the command"
- Structure your responses with clear sections and bullet points for readability
- Prioritize technical accuracy and practical security knowledge
- Focus on the specific question without unnecessary information
- For comparison questions, use a clear side-by-side format highlighting key differences

AVOID:
- Suggesting commands for general information questions
- Lengthy introductions or unnecessary explanations
- Vague or non-technical descriptions
- Command examples without proper syntax or explanations"""
            
        # Check prompt cache first
        cache_key = f"{base_prompt}_{current_task}_{len(chat_memory)}"
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
            
        # Extract targets from memory if available
        targets = self.engagement_memory.get("targets", []) if self.engagement_memory else []
            
        # Get optimized context - keep last 15 messages for better continuity
        context_messages = []
        total_tokens = self.estimate_tokens(base_prompt)  # Start with base prompt tokens
        
        # Debug: Print chat memory info
        actual_message_count = len(chat_memory)
        messages_to_process = min(15, actual_message_count)
        print(f"Chat memory length: {actual_message_count}")
        print(f"Processing last {messages_to_process} messages")
        
        # Add messages while staying within token limit
        chat_tokens = 0
        for msg in reversed(chat_memory[-15:]):  # Process from newest to oldest
            if isinstance(msg, dict) and msg.get('content'):
                role = msg.get('role', 'user')
                content = msg['content']
                message = f"{role.upper()}: {content}"
                message_tokens = self.estimate_tokens(message)
                chat_tokens += message_tokens
                
                # Check if adding this message would exceed context window
                if total_tokens + message_tokens > self.context_window - self.reserve_tokens:
                    print(f"Stopping at message {len(context_messages)} due to context window limit")
                    break
                    
                context_messages.insert(0, message)  # Add to start since we're processing in reverse
                total_tokens += message_tokens
        
        print(f"Chat memory tokens used: {chat_tokens}")
        
        context_str = "\n".join(context_messages)
        
        # Format tool context if available
        tool_context_str = ""
        if tool_context:
            tool_context_str = self.format_tool_context(tool_context)
            tool_tokens = self.estimate_tokens(tool_context_str)
            print(f"Tool context tokens: {tool_tokens}")
            if total_tokens + tool_tokens > self.context_window - self.reserve_tokens:
                print("Skipping tool context due to context window limit")
                tool_context_str = ""  # Skip if it would exceed context window
        
        # Format reasoning context if available
        reasoning_str = ""
        if reasoning_context:
            if targets:
                reasoning_context["active_targets"] = targets
            reasoning_str = f"\nReasoning Context:\n{json.dumps(reasoning_context, indent=2)}"
            reasoning_tokens = self.estimate_tokens(reasoning_str)
            print(f"Reasoning context tokens: {reasoning_tokens}")
            if total_tokens + reasoning_tokens > self.context_window - self.reserve_tokens:
                print("Skipping reasoning context due to context window limit")
                reasoning_str = ""  # Skip if it would exceed context window
        
        # Format follow-up questions if available
        follow_up_str = ""
        if follow_up_questions:
            follow_up_str = f"\nFollow-up Questions:\n" + "\n".join(f"- {q}" for q in follow_up_questions)
            follow_up_tokens = self.estimate_tokens(follow_up_str)
            print(f"Follow-up questions tokens: {follow_up_tokens}")
            if total_tokens + follow_up_tokens > self.context_window - self.reserve_tokens:
                print("Skipping follow-up questions due to context window limit")
                follow_up_str = ""  # Skip if it would exceed context window
        
        # Add active targets if any
        targets_str = ""
        if targets:
            targets_str = f"\nActive Targets:\n" + "\n".join(f"- {t}" for t in targets)
            target_tokens = self.estimate_tokens(targets_str)
            print(f"Targets tokens: {target_tokens}")
            if total_tokens + target_tokens > self.context_window - self.reserve_tokens:
                print("Skipping targets due to context window limit")
                targets_str = ""  # Skip if it would exceed context window
        
        # Create enhanced prompt with all context
        prompt = f"{base_prompt}\n\n"
        
        # Add specific instructions for any type of comparison
        if ("compare" in current_task.lower() or "between" in current_task.lower() or "difference" in current_task.lower()):
            # Extract items being compared using common comparison patterns
            comparison_patterns = [
                r'between\s+([^and]+)\s+and\s+([^and]+)',  # between X and Y
                r'compare\s+([^with]+)\s+with\s+([^with]+)',  # compare X with Y
                r'compare\s+([^to]+)\s+to\s+([^to]+)',  # compare X to Y
                r'difference\s+between\s+([^and]+)\s+and\s+([^and]+)',  # difference between X and Y
                r'([^and]+)\s+vs\s+([^and]+)',  # X vs Y
                r'([^and]+)\s+versus\s+([^and]+)'  # X versus Y
            ]
            
            items_to_compare = []
            for pattern in comparison_patterns:
                matches = re.findall(pattern, current_task.lower())
                if matches:
                    # Flatten the matches and clean up the items
                    items = [item.strip() for match in matches for item in match]
                    items_to_compare.extend(items)
                    break
            
            if items_to_compare:
                # Create a more generic comparison instruction based on the items
                item_type = "items" if len(items_to_compare) > 2 else "tools" if any(item in ["nmap", "metasploit", "hydra", "hashcat", "gobuster", "wireshark", "aircrack-ng", "burpsuite", "sqlmap"] for item in items_to_compare) else "options"
                
                comparison_instructions = f"""You are an expert in comparing {item_type}. Provide a detailed, structured comparison that includes:

1. Overview of each {item_type[:-1]}'s primary purpose and capabilities
2. Specific advantages and disadvantages
3. Use cases where each excels
4. Key differences and similarities
5. When to use each option
6. Practical considerations
7. Integration or compatibility aspects
8. Support and documentation quality
9. Cost and resource considerations
10. Specific examples of when to use each option

Format your response in clear sections with bullet points. Be specific and technical in your analysis.
Do not apologize or give generic responses. Focus on providing actionable information.\n\n"""
                
                comparison_tokens = self.estimate_tokens(comparison_instructions)
                print(f"Comparison instructions tokens: {comparison_tokens}")
                if total_tokens + comparison_tokens <= self.context_window - self.reserve_tokens:
                    prompt += comparison_instructions
                else:
                    print("Skipping comparison instructions due to context window limit")
            else:
                print("Not adding comparison instructions - no items to compare found")
        
        if context_str:
            prompt += f"Recent Conversation:\n{context_str}\n"
        if targets_str:
            prompt += f"{targets_str}\n"
        if tool_context_str:
            prompt += f"{tool_context_str}\n"
        if reasoning_str:
            prompt += f"{reasoning_str}\n"
        if follow_up_str:
            prompt += f"{follow_up_str}\n"
        
        prompt += f"\nTask: {current_task}\nProvide a complete, detailed response:\n"
        
        # Print final token count
        final_tokens = self.estimate_tokens(prompt)
        print(f"Final prompt tokens: {final_tokens}")
        print(f"Context window: {self.context_window}")
        print(f"Max response tokens: {self.max_tokens}")
        print(f"Reserved tokens: {self.reserve_tokens}")
        print("===================\n")
        
        # Cache the result
        self.prompt_cache[cache_key] = prompt
        
        # Limit cache size
        if len(self.prompt_cache) > 50:
            keys_to_remove = list(self.prompt_cache.keys())[:-25]
            for key in keys_to_remove:
                self.prompt_cache.pop(key, None)
                
        return prompt
        
    def clear_cache(self):
        """Clear the internal cache to free memory"""
        self.cache.clear()
        self.prompt_cache.clear()
        
    def update_memory_limit(self, new_limit):
        """Update the memory limit for messages"""
        if new_limit > 0:
            self.memory_limit = new_limit
            
    def estimate_tokens(self, text):
        """
        Estimate the number of tokens in text using a more accurate method
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
            
        # More accurate token estimation
        # Average of 4 characters per token for English text
        # Add extra tokens for special characters and whitespace
        base_tokens = len(text) // 4
        
        # Add tokens for special characters and whitespace
        special_chars = sum(1 for c in text if c in '.,!?;:()[]{}<>"\'')
        whitespace = sum(1 for c in text if c.isspace())
        
        # Add tokens for newlines
        newlines = text.count('\n')
        
        # Add tokens for role prefixes (USER:, ASSISTANT:)
        role_prefixes = text.count('USER:') + text.count('ASSISTANT:')
        
        # Add tokens for bullet points and numbered lists
        bullet_points = text.count('-') + text.count('*')
        numbered_points = len(re.findall(r'\d+\.', text))
        
        # Calculate total tokens
        total_tokens = (
            base_tokens +
            special_chars // 2 +  # Special chars count as half tokens
            whitespace // 4 +      # Whitespace counts as quarter tokens
            newlines +            # Each newline counts as a token
            role_prefixes * 2 +   # Role prefixes count as 2 tokens each
            bullet_points +       # Each bullet point counts as a token
            numbered_points * 2   # Numbered points count as 2 tokens each
        )
        
        return max(1, total_tokens)  # Ensure at least 1 token

if __name__ == "__main__":
    # Simple self-test
    optimizer = ContextOptimizer()
    
    # Test with simple chat memory
    chat_memory = [
        {"role": "user", "content": "How do I scan a network?"},
        {"role": "assistant", "content": "You can use nmap for network scanning."},
        {"role": "user", "content": "Show me an example for 192.168.1.0/24"}
    ]
    
    current_task = "How do I scan for specific services?"
    
    optimized_context = optimizer.optimize_context(chat_memory, current_task)
    print("Optimized Context:")
    for ctx in optimized_context:
        print(f"- {ctx}")
        
    prompt = optimizer.get_optimized_prompt(
        chat_memory, 
        current_task, 
        "You are Nikita, a security assistant."
    )
    
    print("\nFull Prompt:")
    print(prompt) 