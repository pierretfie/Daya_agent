#!/usr/bin/env python3
"""
Documentation Verifier Module for Daya Agent

Handles documentation verification, citation tracking, and local knowledge base management.
"""

import json
import os
from pathlib import Path
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from rich.console import Console

console = Console()

class DocumentationVerifier:
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the documentation verifier.
        
        Args:
            knowledge_base_path (str, optional): Path to local knowledge base directory
        """
        self.knowledge_base_path = knowledge_base_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "knowledge_base"
        )
        self.citation_cache = {}
        self.verification_cache = {}
        
        # Create knowledge base directory if it doesn't exist
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        # Initialize verification sources
        self.verification_sources = {
            "man_pages": self._verify_man_page,
            "official_docs": self._verify_official_docs,
            "security_advisories": self._verify_security_advisories,
            "community_sources": self._verify_community_sources
        }

    def verify_tool_documentation(self, tool_name: str, documentation: Dict) -> Tuple[bool, Dict]:
        """
        Verify tool documentation against multiple sources.
        
        Args:
            tool_name (str): Name of the tool
            documentation (Dict): Documentation to verify
            
        Returns:
            Tuple[bool, Dict]: (is_verified, verification_details)
        """
        # Check cache first
        cache_key = hashlib.md5(f"{tool_name}_{json.dumps(documentation)}".encode()).hexdigest()
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
            
        verification_results = {
            "sources_checked": [],
            "matches": [],
            "discrepancies": [],
            "confidence_score": 0.0,
            "last_verified": datetime.now().isoformat()
        }
        
        # Check each verification source
        for source_name, verifier in self.verification_sources.items():
            try:
                is_verified, details = verifier(tool_name, documentation)
                verification_results["sources_checked"].append(source_name)
                if is_verified:
                    verification_results["matches"].append({
                        "source": source_name,
                        "details": details
                    })
                    verification_results["confidence_score"] += 0.25  # Each source adds 0.25 to confidence
                else:
                    verification_results["discrepancies"].append({
                        "source": source_name,
                        "details": details
                    })
            except Exception as e:
                console.print(f"[yellow]Warning: Verification failed for {source_name}: {str(e)}[/yellow]")
        
        # Cache the results
        self.verification_cache[cache_key] = (
            verification_results["confidence_score"] >= 0.5,  # Consider verified if 2+ sources match
            verification_results
        )
        
        return self.verification_cache[cache_key]

    def _verify_man_page(self, tool_name: str, documentation: Dict) -> Tuple[bool, Dict]:
        """Verify documentation against man page content"""
        try:
            # Get man page content
            man_page = self._get_man_page(tool_name)
            if not man_page:
                return False, {"error": "Man page not found"}
            
            # Compare key sections
            matches = []
            discrepancies = []
            
            # Check purpose/description
            if "purpose" in documentation:
                if documentation["purpose"].lower() in man_page.lower():
                    matches.append("purpose")
                else:
                    discrepancies.append("purpose")
            
            # Check syntax
            if "syntax" in documentation:
                if documentation["syntax"].lower() in man_page.lower():
                    matches.append("syntax")
                else:
                    discrepancies.append("syntax")
            
            # Check parameters
            if "parameters" in documentation:
                param_matches = 0
                for param in documentation["parameters"]:
                    if param.lower() in man_page.lower():
                        param_matches += 1
                if param_matches / len(documentation["parameters"]) >= 0.8:
                    matches.append("parameters")
                else:
                    discrepancies.append("parameters")
            
            return len(matches) >= 2, {
                "matches": matches,
                "discrepancies": discrepancies
            }
        except Exception as e:
            return False, {"error": str(e)}

    def _verify_official_docs(self, tool_name: str, documentation: Dict) -> Tuple[bool, Dict]:
        """Verify documentation against official documentation sources"""
        try:
            # List of official documentation URLs for common tools
            official_docs = {
                "nmap": "https://nmap.org/book/man.html",
                "metasploit": "https://docs.metasploit.com/",
                "wireshark": "https://www.wireshark.org/docs/",
                "burpsuite": "https://portswigger.net/burp/documentation",
                "sqlmap": "https://github.com/sqlmapproject/sqlmap/wiki"
            }
            
            if tool_name.lower() not in official_docs:
                return False, {"error": "No official documentation URL found"}
            
            # Fetch official documentation
            response = requests.get(official_docs[tool_name.lower()])
            if response.status_code != 200:
                return False, {"error": f"Failed to fetch official docs: {response.status_code}"}
            
            # Compare key sections
            matches = []
            discrepancies = []
            
            # Check purpose/description
            if "purpose" in documentation:
                if documentation["purpose"].lower() in response.text.lower():
                    matches.append("purpose")
                else:
                    discrepancies.append("purpose")
            
            # Check syntax
            if "syntax" in documentation:
                if documentation["syntax"].lower() in response.text.lower():
                    matches.append("syntax")
                else:
                    discrepancies.append("syntax")
            
            return len(matches) >= 1, {
                "matches": matches,
                "discrepancies": discrepancies,
                "source_url": official_docs[tool_name.lower()]
            }
        except Exception as e:
            return False, {"error": str(e)}

    def _verify_security_advisories(self, tool_name: str, documentation: Dict) -> Tuple[bool, Dict]:
        """Verify documentation against security advisories"""
        try:
            # List of security advisory sources
            advisory_sources = [
                "https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=" + tool_name,
                "https://nvd.nist.gov/vuln/search/results?form_type=Basic&results_type=overview&query=" + tool_name,
                "https://www.exploit-db.com/search?text=" + tool_name
            ]
            
            matches = []
            discrepancies = []
            
            for source in advisory_sources:
                response = requests.get(source)
                if response.status_code == 200:
                    # Check if tool is mentioned in security context
                    if tool_name.lower() in response.text.lower():
                        matches.append(source)
                    else:
                        discrepancies.append(source)
            
            return len(matches) >= 1, {
                "matches": matches,
                "discrepancies": discrepancies
            }
        except Exception as e:
            return False, {"error": str(e)}

    def _verify_community_sources(self, tool_name: str, documentation: Dict) -> Tuple[bool, Dict]:
        """Verify documentation against community sources"""
        try:
            # List of community sources
            community_sources = [
                f"https://github.com/search?q={tool_name}&type=repositories",
                f"https://stackoverflow.com/search?q={tool_name}",
                f"https://www.reddit.com/search/?q={tool_name}"
            ]
            
            matches = []
            discrepancies = []
            
            for source in community_sources:
                response = requests.get(source)
                if response.status_code == 200:
                    # Check if tool is mentioned in relevant context
                    if tool_name.lower() in response.text.lower():
                        matches.append(source)
                    else:
                        discrepancies.append(source)
            
            return len(matches) >= 2, {
                "matches": matches,
                "discrepancies": discrepancies
            }
        except Exception as e:
            return False, {"error": str(e)}

    def _get_man_page(self, tool_name: str) -> Optional[str]:
        """Get man page content for a tool"""
        try:
            import subprocess
            result = subprocess.run(['man', tool_name], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def update_local_knowledge_base(self, tool_name: str, documentation: Dict, verification_results: Dict):
        """
        Update the local knowledge base with verified documentation.
        
        Args:
            tool_name (str): Name of the tool
            documentation (Dict): Documentation to store
            verification_results (Dict): Verification results
        """
        try:
            # Create tool directory if it doesn't exist
            tool_dir = os.path.join(self.knowledge_base_path, tool_name)
            os.makedirs(tool_dir, exist_ok=True)
            
            # Save documentation
            doc_path = os.path.join(tool_dir, "documentation.json")
            with open(doc_path, 'w') as f:
                json.dump({
                    "tool_name": tool_name,
                    "documentation": documentation,
                    "verification": verification_results,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            
            # Save citations
            citations_path = os.path.join(tool_dir, "citations.json")
            citations = {
                "official_docs": verification_results.get("matches", []),
                "community_sources": verification_results.get("discrepancies", []),
                "last_updated": datetime.now().isoformat()
            }
            with open(citations_path, 'w') as f:
                json.dump(citations, f, indent=2)
                
            return True
        except Exception as e:
            console.print(f"[red]Error updating knowledge base: {str(e)}[/red]")
            return False

    def get_local_documentation(self, tool_name: str) -> Optional[Dict]:
        """
        Get verified documentation from local knowledge base.
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[Dict]: Documentation if found, None otherwise
        """
        try:
            doc_path = os.path.join(self.knowledge_base_path, tool_name, "documentation.json")
            if os.path.exists(doc_path):
                with open(doc_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load local documentation: {str(e)}[/yellow]")
            return None

    def get_citations(self, tool_name: str) -> Optional[Dict]:
        """
        Get citations for tool documentation.
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[Dict]: Citations if found, None otherwise
        """
        try:
            citations_path = os.path.join(self.knowledge_base_path, tool_name, "citations.json")
            if os.path.exists(citations_path):
                with open(citations_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load citations: {str(e)}[/yellow]")
            return None 