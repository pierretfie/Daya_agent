from typing import Optional, Dict
import re

class CommandProcessor:
    def process_command(self, command: str, context: Optional[Dict] = None) -> Dict:
        """Process a command with enhanced flexibility for starting words"""
        result = {
            "success": False,
            "output": "",
            "error": None,
            "command_type": None
        }
        
        # Normalize the command
        command = command.strip().lower()
        
        # Enhanced command pattern matching
        command_patterns = {
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
            security_keywords = ["scan", "check", "analyze", "test", "verify", "examine"]
            if any(keyword in command for keyword in security_keywords):
                result["command_type"] = "security_analysis"
                result.update(self._process_security_analysis(command))
                return result
            
            # Check for information request keywords
            info_keywords = ["what", "which", "how", "tell", "show", "list", "get", "find"]
            if any(keyword in command for keyword in info_keywords):
                result["command_type"] = "information_request"
                result.update(self._process_information_request(command))
                return result
        
        # If still no match, return error
        result["error"] = "Command not recognized. Please try rephrasing your request."
        return result 