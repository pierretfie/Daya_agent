from typing import Optional, Dict
import re

class CommandProcessor:
    def __init__(self):
        self.security_keywords = ["exploit", "hack", "attack", "penetrate", "scan", "check", "analyze", "test", "verify", "examine"]
        self.info_keywords = ["what", "which", "how", "tell", "show", "list", "get", "find"]
        
    def _is_valid_target(self, target: str) -> bool:
        """Validate if the target is in a valid format"""
        # Check for IP address
        if re.match(r'^(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?$', target):
            return True
        # Check for hostname
        if re.match(r'^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$', target):
            return True
        return False

    def process_command(self, command: str, context: Optional[Dict] = None) -> Dict:
        """Process a command with enhanced flexibility for starting words"""
        result = {
            "success": False,
            "output": "",
            "error": None,
            "command_type": None,
            "requires_confirmation": False,
            "can_continue": True,  # New flag to indicate if model can continue generating
            "suggested_follow_up": None  # New field for suggested follow-up content
        }
        
        # Normalize the command
        command = command.strip().lower()
        
        # Enhanced command pattern matching
        command_patterns = {
            # Security-related patterns
            r'(?:exploit|attack|hack|penetrate)\s+(?:port|service)\s+(?:on|at)\s+([a-zA-Z0-9\./-]+)': "security_alert",
            r'(?:exploit|attack|hack)\s+([a-zA-Z0-9\./-]+)\s+(?:on|at)\s+(?:port|service)': "security_alert",
            
            # Security scan patterns
            r'(?:scan|check|analyze|examine)\s+(?:port|ports|service|services)\s+(?:on|for|at)\s+([a-zA-Z0-9\./-]+)': "port_scan",
            r'(?:scan|check|analyze)\s+([a-zA-Z0-9\./-]+)\s+(?:for|on)\s+(?:port|ports|service|services)': "port_scan",
            
            # Network analysis patterns
            r'(?:analyze|check|examine)\s+(?:network|connection|connectivity)\s+(?:for|on|at)\s+([a-zA-Z0-9\./-]+)': "network_analysis",
            r'(?:check|verify|test)\s+(?:if|whether)\s+([a-zA-Z0-9\./-]+)\s+(?:is|are)\s+(?:up|down|online|offline)': "network_check",
            
            # Information gathering patterns
            r'(?:get|find|show|list)\s+(?:info|information|details)\s+(?:about|on|for)\s+([a-zA-Z0-9\./-]+)': "info_gathering",
            r'(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:details|info|information)\s+(?:about|on|for)\s+([a-zA-Z0-9\./-]+)': "info_gathering",
            
            # System check patterns
            r'(?:check|verify|test)\s+(?:system|machine|device)\s+(?:status|state|condition)\s+(?:of|for|on)\s+([a-zA-Z0-9\./-]+)': "system_check",
            r'(?:is|are)\s+([a-zA-Z0-9\./-]+)\s+(?:up|down|online|offline|running|stopped)': "system_check"
        }
        
        # Try to match the command against patterns
        for pattern, command_type in command_patterns.items():
            match = re.search(pattern, command)
            if match:
                result["command_type"] = command_type
                target = match.group(1)
                
                # Validate target
                if not self._is_valid_target(target):
                    result["error"] = f"Invalid target: {target}"
                    result["suggested_follow_up"] = "Please provide a valid IP address or hostname."
                    return result
                
                # Handle security-related commands with extra caution
                if command_type == "security_alert":
                    result["requires_confirmation"] = True
                    result["output"] = f"⚠️ Security Alert: This command appears to be attempting to exploit {target}. Such actions may be illegal and unethical. Please confirm if you have proper authorization to perform this action."
                    result["suggested_follow_up"] = "Would you like to know more about ethical security testing practices?"
                    return result
                
                # Process based on command type
                if command_type == "port_scan":
                    result.update(self._process_port_scan(target))
                elif command_type == "network_analysis":
                    result.update(self._process_network_analysis(target))
                elif command_type == "info_gathering":
                    result.update(self._process_info_gathering(target))
                elif command_type == "system_check":
                    result.update(self._process_system_check(target))
                
                return result
        
        # If no pattern matched, try semantic analysis
        if not result["command_type"]:
            # Check for security-related keywords
            if any(keyword in command for keyword in self.security_keywords):
                result["command_type"] = "security_analysis"
                result.update(self._process_security_analysis(command))
                return result
            
            # Check for information request keywords
            if any(keyword in command for keyword in self.info_keywords):
                result["command_type"] = "information_request"
                result.update(self._process_information_request(command))
                return result
        
        # If still no match, return error with suggestion
        result["error"] = "Command not recognized. Please try rephrasing your request."
        result["suggested_follow_up"] = "Would you like me to help you formulate the correct command?"
        return result

    def _process_security_analysis(self, command: str) -> Dict:
        """Process security-related commands with enhanced safety checks"""
        result = {
            "success": False,
            "output": "",
            "error": None,
            "requires_confirmation": True,
            "can_continue": True,
            "suggested_follow_up": None
        }
        
        # Extract potential target
        target_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?', command)
        target = target_match.group(0) if target_match else "unknown target"
        
        # Check for potentially harmful commands
        if any(word in command for word in ["exploit", "hack", "attack", "penetrate"]):
            result["output"] = f"⚠️ Security Alert: This command appears to be attempting to exploit {target}. Such actions may be illegal and unethical. Please confirm if you have proper authorization to perform this action."
            result["suggested_follow_up"] = "Would you like to learn about legal and ethical security testing practices?"
            return result
        
        # For non-exploitative security commands
        result["output"] = f"Security analysis requested for {target}. Please confirm if you have proper authorization to perform this action."
        result["suggested_follow_up"] = "Would you like to know more about security best practices?"
        return result

    def _process_port_scan(self, target: str) -> Dict:
        """Process port scan commands"""
        return {
            "success": True,
            "output": f"Port scan requested for {target}. Please confirm if you have proper authorization to perform this action.",
            "requires_confirmation": True,
            "can_continue": True,
            "suggested_follow_up": "Would you like to learn about different types of port scanning techniques?"
        }

    def _process_network_analysis(self, target: str) -> Dict:
        """Process network analysis commands"""
        return {
            "success": True,
            "output": f"Network analysis requested for {target}. Please confirm if you have proper authorization to perform this action.",
            "requires_confirmation": True,
            "can_continue": True,
            "suggested_follow_up": "Would you like to know more about network analysis tools and techniques?"
        }

    def _process_info_gathering(self, target: str) -> Dict:
        """Process information gathering commands"""
        return {
            "success": True,
            "output": f"Information gathering requested for {target}. Please confirm if you have proper authorization to perform this action.",
            "requires_confirmation": True,
            "can_continue": True,
            "suggested_follow_up": "Would you like to learn about legal information gathering methods?"
        }

    def _process_system_check(self, target: str) -> Dict:
        """Process system check commands"""
        return {
            "success": True,
            "output": f"System check requested for {target}. Please confirm if you have proper authorization to perform this action.",
            "requires_confirmation": True,
            "can_continue": True,
            "suggested_follow_up": "Would you like to know more about system monitoring and diagnostics?"
        }

    def _process_information_request(self, command: str) -> Dict:
        """Process information request commands"""
        return {
            "success": True,
            "output": "Information request received. Please provide more specific details about what information you need.",
            "requires_confirmation": False,
            "can_continue": True,
            "suggested_follow_up": "Would you like me to help you formulate a more specific question?"
        } 