#!/usr/bin/env python3
"""
Semantic Context Optimizer Module for Daya Agent

Provides an additional layer of context optimization by analyzing the semantic
relevance of messages, prioritizing content based on meaning rather than just
token counts, and improving context quality through semantic clustering.
"""

import re
import numpy as np
from datetime import datetime
import json
from collections import defaultdict
import os

class SemanticContextOptimizer:
    """
    Enhances context optimization by analyzing semantic relevance of messages
    and improving context quality through semantic clustering and prioritization.
    
    This layer works on top of the base ContextOptimizer to provide more
    sophisticated context selection based on meaning rather than just tokens.
    """
    
    def __init__(self, base_optimizer=None):
        """
        Initialize the semantic context optimizer.
        
        Args:
            base_optimizer: The base ContextOptimizer instance to enhance
        """
        self.base_optimizer = base_optimizer
        self.semantic_cache = {}
        self.topic_clusters = {}
        self.relevance_scores = {}
        self.keyword_index = defaultdict(list)
        self.semantic_memory = {}
        
        # Load semantic patterns if available
        self.patterns_file = os.path.join(os.path.dirname(__file__), "human_like_patterns.json")
        self.semantic_patterns = self._load_patterns()
        
    def _load_patterns(self):
        """Load semantic patterns from file"""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading semantic patterns: {e}")
            return {}
    
    def extract_keywords(self, text):
        """
        Extract important keywords from text for semantic indexing.
        
        Args:
            text (str): Text to extract keywords from
            
        Returns:
            list: List of extracted keywords
        """
        if not text:
            return []
            
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stopwords
        stopwords = {
            'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'with', 'on', 
            'that', 'this', 'it', 'as', 'be', 'by', 'are', 'was', 'were', 
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'am', 'is', 'are',
            'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Add security-specific terms with higher weight (represented by repetition)
        security_terms = {
            'vulnerability', 'exploit', 'attack', 'security', 'threat', 'malware',
            'virus', 'trojan', 'ransomware', 'phishing', 'breach', 'hack', 'backdoor',
            'encryption', 'firewall', 'authentication', 'authorization', 'mitigation',
            'patch', 'cve', 'pentest', 'penetration', 'scan', 'reconnaissance'
        }
        
        # Add security terms with higher weight by duplicating them
        for word in keywords:
            if word in security_terms:
                keywords.append(word)  # Add again to increase weight
                
        return keywords
    
    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two text snippets.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Extract keywords from both texts
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def cluster_by_topic(self, messages):
        """
        Cluster messages by topic for better context organization.
        NOTE: This function is currently disabled as it was causing issues with responses.
        Instead, we're using a simpler approach that preserves the original conversation flow.
        
        Args:
            messages (list): List of message dictionaries
            
        Returns:
            dict: Messages clustered by topic (currently returns empty dict)
        """
        # DISABLED: Return empty dictionary to prevent topic clustering
        return {}
        
        # The original implementation is commented out below
        '''
        if not messages:
            return {}
            
        clusters = defaultdict(list)
        
        # First pass: identify potential topics
        topics = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or not msg.get('content'):
                continue
                
            content = msg['content']
            
            # Extract potential topic from message
            # Look for key phrases that might indicate a topic
            topic_indicators = [
                (r'(?:about|regarding|concerning)\s+(\w+(?:\s+\w+){0,3})', 1),  # "about X"
                (r'(?:how to|how do I)\s+(\w+(?:\s+\w+){0,3})', 1),  # "how to X"
                (r'^(\w+(?:\s+\w+){0,2})\s+(?:is|are|means)', 1),  # "X is/are/means"
                (r'(?:what is|what are)\s+(\w+(?:\s+\w+){0,3})', 1),  # "what is X"
                (r'(?:using|with)\s+(\w+(?:\s+\w+){0,2})', 1)  # "using X"
            ]
            
            potential_topic = None
            for pattern, group in topic_indicators:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    potential_topic = match.group(group).strip().lower()
                    break
                    
            # If no topic found, use the first few words
            if not potential_topic and len(content.split()) > 3:
                potential_topic = ' '.join(content.split()[:3]).lower()
                
            # Add to topics if found
            if potential_topic:
                topics.append((i, potential_topic))
                
        # Second pass: assign messages to topics based on similarity
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or not msg.get('content'):
                continue
                
            content = msg['content']
            best_topic = None
            best_score = 0.3  # Threshold for topic assignment
            
            for topic_idx, topic in topics:
                if topic_idx == i:  # Skip comparing to self
                    continue
                    
                score = self.calculate_semantic_similarity(content, topic)
                if score > best_score:
                    best_score = score
                    best_topic = topic
                    
            # Assign to best topic or create a new one
            if best_topic:
                clusters[best_topic].append(msg)
            else:
                # Create a new topic from this message
                new_topic = ' '.join(content.split()[:3]).lower()
                clusters[new_topic].append(msg)
                
        return dict(clusters)
        '''
    
    def prioritize_by_relevance(self, messages, current_task, targets=None):
        """
        Prioritize messages by relevance to the current task.
        
        Args:
            messages (list): List of message dictionaries
            current_task (str): Current user task/query
            targets (list, optional): List of targets to prioritize
            
        Returns:
            list: Messages sorted by relevance
        """
        if not messages or not current_task:
            return messages
            
        # Calculate relevance scores for each message
        scored_messages = []
        
        for msg in messages:
            if not isinstance(msg, dict) or not msg.get('content'):
                continue
                
            content = msg['content']
            
            # Calculate base similarity score
            similarity = self.calculate_semantic_similarity(content, current_task)
            
            # Adjust score based on additional factors
            score = similarity
            
            # Boost score for messages containing targets
            if targets:
                for target in targets:
                    if target.lower() in content.lower():
                        score += 0.2
                        break
                        
            # Boost score for recent messages (recency bias)
            if msg.get('timestamp'):
                try:
                    msg_time = datetime.strptime(msg['timestamp'], "%Y-%m-%d %H:%M:%S")
                    now = datetime.now()
                    hours_ago = (now - msg_time).total_seconds() / 3600
                    
                    # Apply recency boost (diminishing with time)
                    if hours_ago < 1:
                        score += 0.3
                    elif hours_ago < 24:
                        score += 0.1
                except:
                    pass
                    
            # Boost score for assistant responses that might contain valuable info
            if msg.get('role') == 'assistant':
                # Check for code blocks, command examples, or explanations
                if '```' in content or '`' in content:
                    score += 0.2
                if re.search(r'(explanation|explained|means|definition|defined as)', content, re.IGNORECASE):
                    score += 0.1
                    
            scored_messages.append((msg, score))
            
        # Sort by score in descending order
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted messages without scores
        return [msg for msg, _ in scored_messages]
    
    def extract_entities(self, text):
        """
        Extract named entities from text for better context understanding.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            dict: Dictionary of extracted entities by type
        """
        if not text:
            return {}
            
        entities = {
            'ip_addresses': [],
            'domains': [],
            'commands': [],
            'tools': [],
            'cves': []
        }
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'
        entities['ip_addresses'] = re.findall(ip_pattern, text)
        
        # Extract domains
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        entities['domains'] = re.findall(domain_pattern, text)
        
        # Extract commands (text between backticks or after command prompts)
        command_patterns = [
            r'`([^`]+)`',  # Text between backticks
            r'\$\s*([^;\n]+)',  # Text after $ prompt
            r'#\s*([^;\n]+)'  # Text after # prompt
        ]
        
        for pattern in command_patterns:
            entities['commands'].extend(re.findall(pattern, text))
            
        # Extract security tools
        security_tools = [
            'nmap', 'metasploit', 'hydra', 'hashcat', 'gobuster', 'wireshark',
            'aircrack-ng', 'burpsuite', 'sqlmap', 'nikto', 'dirb', 'netcat',
            'tcpdump', 'john', 'snort', 'openvas', 'masscan', 'wpscan'
        ]
        
        for tool in security_tools:
            if re.search(r'\b' + re.escape(tool) + r'\b', text, re.IGNORECASE):
                entities['tools'].append(tool)
                
        # Extract CVEs
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        entities['cves'] = re.findall(cve_pattern, text, re.IGNORECASE)
        
        return entities
    
    def optimize_context(self, chat_memory, current_task, targets=None):
        """
        Enhance context optimization with semantic analysis.
        
        Args:
            chat_memory (list): List of chat messages
            current_task (str): Current user task/query
            targets (list, optional): List of targets to prioritize
            
        Returns:
            list: Optimized context messages
        """
        # Check cache first for performance
        cache_key = f"semantic_{current_task}_{len(chat_memory)}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
            
        # If base optimizer is available, get its optimized context first
        if self.base_optimizer:
            base_context = self.base_optimizer.optimize_context(chat_memory, current_task, targets)
            
            # If base context is very small, just return it
            if len(base_context) <= 5:
                return base_context
        else:
            # Use the last 15 messages as base context if no base optimizer
            base_context = [msg['content'] for msg in chat_memory[-15:] 
                           if isinstance(msg, dict) and msg.get('content')]
                           
        # Extract entities from current task
        task_entities = self.extract_entities(current_task)
        
        # Prioritize messages by semantic relevance
        prioritized_messages = self.prioritize_by_relevance(chat_memory, current_task, targets)
        
        # Get the most relevant messages (up to 10)
        relevant_msgs = [msg['content'] for msg in prioritized_messages[:10]
                         if isinstance(msg, dict) and msg.get('content')]
                         
        # Ensure we have the most recent message for continuity
        if chat_memory and isinstance(chat_memory[-1], dict) and chat_memory[-1].get('content'):
            last_msg = chat_memory[-1]['content']
            if last_msg not in relevant_msgs:
                relevant_msgs.append(last_msg)
                
        # Cache the result
        self.semantic_cache[cache_key] = relevant_msgs
        
        # Limit cache size
        if len(self.semantic_cache) > 50:
            keys_to_remove = list(self.semantic_cache.keys())[:-25]
            for key in keys_to_remove:
                self.semantic_cache.pop(key, None)
                
        return relevant_msgs
        
        """
        Get a semantically optimized prompt with enhanced context for the LLM.
        
        The optimized prompt is generated by first getting the base optimized prompt
        from the base optimizer. Then, the method enhances specific parts of the prompt
        such as the conversation context, reasoning context and task description.
        
        The conversation context is enhanced by clustering similar messages together
        and adding topic headers. The reasoning context is enhanced by extracting entities
        from the current task and adding them to the reasoning context. The task description
        is also enhanced by adding follow-up questions and tool-specific context.
        
        Args:
            chat_memory (list): List of chat messages
            current_task (str): Current user task/query
            base_prompt (str): Base system prompt
            reasoning_context (dict, optional): Additional reasoning context
            follow_up_questions (list, optional): Follow-up questions to include
            tool_context (dict, optional): Tool-specific context
            
        Returns:
            str: Optimized prompt
        """
            
    def get_optimized_prompt(self, chat_memory, current_task, base_prompt, reasoning_context=None,
                            follow_up_questions=None, tool_context=None):
        """
        Generate an optimized prompt with enhanced semantic understanding.
        
        Args:
            chat_memory (list): List of chat messages
            current_task (str): Current user task/query
            base_prompt (str): Base prompt template
            reasoning_context (dict, optional): Reasoning context data
            follow_up_questions (list, optional): Follow-up questions
            tool_context (dict, optional): Context for tool usage
            
        Returns:
            str: Semantically optimized prompt
        """
        # If base optimizer is available, get its optimized prompt first
        if self.base_optimizer:
            base_optimized_prompt = self.base_optimizer.get_optimized_prompt(
                chat_memory, current_task, base_prompt, reasoning_context,
                follow_up_questions, tool_context
            )
            
            # IMPORTANT: Don't trim or restructure the prompt as it was causing issues
            # Return the base optimized prompt without semantic modifications
            return base_optimized_prompt
            
            # Extract the components from the base optimized prompt
            prompt_parts = base_optimized_prompt.split("\n\n")
            
            # The base prompt is usually the first part
            enhanced_base = prompt_parts[0]
            
            # Extract other parts
            conversation_part = ""
            tool_part = ""
            reasoning_part = ""
            follow_up_part = ""
            task_part = ""
            
            for part in prompt_parts[1:]:
                if part.startswith("Recent Conversation:"):
                    conversation_part = part
                elif part.startswith("Tool Documentation:") or part.startswith("Common Use Cases:"):
                    tool_part = part
                elif part.startswith("Reasoning Context:"):
                    reasoning_part = part
                elif part.startswith("Follow-up Questions:"):
                    follow_up_part = part
                elif part.startswith("Task:"):
                    task_part = part
                    
            # Now we can enhance specific parts
            
            # Enhance the conversation part with semantic clustering
            if conversation_part:
                # Extract the original messages
                messages = []
                for line in conversation_part.split("\n")[1:]:  # Skip the "Recent Conversation:" header
                    if line.startswith(("USER:", "ASSISTANT:")):
                        role, content = line.split(":", 1)
                        messages.append({"role": role.lower(), "content": content.strip()})
                
                # DISABLED: Cluster by topic - this was causing issues with responses
                # Instead, keep the original conversation format without clustering
                new_conversation = ["Recent Conversation:"]
                for msg in messages:
                    new_conversation.append(f"{msg['role'].upper()}: {msg['content']}")
                
                conversation_part = "\n".join(new_conversation)
            
            # Enhance reasoning context with entity extraction
            if reasoning_part and current_task:
                entities = self.extract_entities(current_task)
                if entities and any(entities.values()):
                    # Convert the reasoning part to a dictionary
                    try:
                        reasoning_dict = json.loads(reasoning_part.replace("Reasoning Context:\n", ""))
                        # Add entities to reasoning
                        reasoning_dict["extracted_entities"] = entities
                        # Convert back to string
                        reasoning_part = f"Reasoning Context:\n{json.dumps(reasoning_dict, indent=2)}"
                    except:
                        # If parsing fails, just append entities
                        entities_str = json.dumps(entities, indent=2)
                        reasoning_part += f"\nExtracted Entities:\n{entities_str}"
            
            # Rebuild the prompt with enhanced parts
            enhanced_prompt = enhanced_base + "\n\n"
            
            if conversation_part:
                enhanced_prompt += conversation_part + "\n\n"
            if tool_part:
                enhanced_prompt += tool_part + "\n\n"
            if reasoning_part:
                enhanced_prompt += reasoning_part + "\n\n"
            if follow_up_part:
                enhanced_prompt += follow_up_part + "\n\n"
            if task_part:
                enhanced_prompt += task_part + "\n"
                
            return enhanced_prompt
        else:
            # If no base optimizer, create a simple optimized prompt
            prompt = f"{base_prompt}\n\n"
            
            # Get semantically optimized context
            context_messages = self.optimize_context(chat_memory, current_task, None)  # No targets passed
            
            if context_messages:
                prompt += f"Recent Conversation:\n"
                for i, msg in enumerate(context_messages):
                    role = "USER" if i % 2 == 0 else "ASSISTANT"
                    prompt += f"{role}: {msg}\n"
                prompt += "\n"
                
            # Add reasoning context if available
            if reasoning_context:
                prompt += f"Reasoning Context:\n{json.dumps(reasoning_context, indent=2)}\n\n"
                
            # Add follow-up questions if available
            if follow_up_questions:
                prompt += f"Follow-up Questions:\n"
                for q in follow_up_questions:
                    prompt += f"- {q}\n"
                prompt += "\n"
                
            # Add tool context if available
            if tool_context:
                prompt += f"Tool Context:\n{json.dumps(tool_context, indent=2)}\n\n"
                
            # Add current task
            prompt += f"Task: {current_task}\nProvide a complete, detailed response:\n"
            
            return prompt
            
    def clear_cache(self):
        """Clear the semantic cache"""
        self.semantic_cache.clear()
        self.topic_clusters.clear()
        self.relevance_scores.clear()
        
    def update_with_feedback(self, user_feedback, last_context):
        """
        Update semantic patterns based on user feedback.
        
        Args:
            user_feedback (str): User feedback on response quality
            last_context (list): Last context used for generation
        """
        # Simple implementation - could be expanded with more sophisticated learning
        if not user_feedback or not last_context:
            return
            
        feedback_lower = user_feedback.lower()
        
        # Check if feedback is positive or negative
        positive_indicators = ['good', 'great', 'excellent', 'helpful', 'useful', 'thanks', 'thank you']
        negative_indicators = ['bad', 'wrong', 'incorrect', 'not helpful', 'useless', 'irrelevant']
        
        is_positive = any(indicator in feedback_lower for indicator in positive_indicators)
        is_negative = any(indicator in feedback_lower for indicator in negative_indicators)
        
        if not is_positive and not is_negative:
            return  # Neutral feedback, no action needed
            
        # Extract keywords from the context
        all_context_text = " ".join(last_context)
        keywords = self.extract_keywords(all_context_text)
        
        # Update semantic memory based on feedback
        for keyword in keywords:
            if keyword not in self.semantic_memory:
                self.semantic_memory[keyword] = {'positive': 0, 'negative': 0, 'total': 0}
                
            self.semantic_memory[keyword]['total'] += 1
            
            if is_positive:
                self.semantic_memory[keyword]['positive'] += 1
            elif is_negative:
                self.semantic_memory[keyword]['negative'] += 1


if __name__ == "__main__":
    # Simple self-test
    from context_optimizer import ContextOptimizer
    
    # Create base optimizer
    base_optimizer = ContextOptimizer()
    
    # Create semantic optimizer
    semantic_optimizer = SemanticContextOptimizer(base_optimizer)
    
    # Test with simple chat memory
    chat_memory = [
        {"role": "user", "content": "How do I scan a network?", "timestamp": "2023-01-01 10:00:00"},
        {"role": "assistant", "content": "You can use nmap for network scanning.", "timestamp": "2023-01-01 10:01:00"},
        {"role": "user", "content": "Show me an example for 192.168.1.0/24", "timestamp": "2023-01-01 10:02:00"},
        {"role": "assistant", "content": "You can use: `nmap -sV 192.168.1.0/24`", "timestamp": "2023-01-01 10:03:00"},
        {"role": "user", "content": "What does -sV do?", "timestamp": "2023-01-01 10:04:00"},
        {"role": "assistant", "content": "The -sV flag in nmap enables version detection.", "timestamp": "2023-01-01 10:05:00"},
        {"role": "user", "content": "How do I check for vulnerabilities?", "timestamp": "2023-01-01 10:06:00"}
    ]
    
    current_task = "I want to scan 10.0.0.0/24 for open ports and vulnerabilities"
    
    # Test semantic optimization
    optimized_context = semantic_optimizer.optimize_context(chat_memory, current_task)
    print("Semantically Optimized Context:")
    for ctx in optimized_context:
        print(f"- {ctx}")
        
    # Test prompt optimization
    prompt = semantic_optimizer.get_optimized_prompt(
        chat_memory, 
        current_task, 
        "You are Daya, a security assistant."
    )
    
    print("\nOptimized Prompt:")
    print(prompt)
