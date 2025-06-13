"""
System configuration for Advanced GRC System
"""
import os
from typing import Dict, Any

def get_system_config() -> Dict[str, Any]:
    """Get complete system configuration."""
    
    # Base paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return {
        "llm": {
            "provider": "gemini",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "4000")),
            "top_p": float(os.getenv("GEMINI_TOP_P", "0.8")),
            "top_k": int(os.getenv("GEMINI_TOP_K", "40")),
            "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
        },
        "data_paths": {
            "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
            "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
        },
        "system": {
            "max_concurrent_agents": int(os.getenv("MAX_CONCURRENT_AGENTS", "10")),
            "analysis_timeout": int(os.getenv("ANALYSIS_TIMEOUT", "300")),
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "performance_monitoring": True,
            "agent_health_checks": True
        },
        "agent_config": {
            "coverage_analyst": {
                "max_concurrent_requests": 3,
                "timeout": 30,
                "retry_attempts": 2
            },
            "gap_identifier": {
                "max_concurrent_requests": 2,
                "timeout": 45,
                "retry_attempts": 2
            },
            "risk_assessor": {
                "max_concurrent_requests": 2,
                "timeout": 60,
                "retry_attempts": 2
            }
        },
        "reporting": {
            "executive_dashboard": True,
            "detailed_reports": True,
            "export_formats": ["json", "pdf", "excel"],
            "auto_recommendations": True
        }
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate system configuration."""
    
    required_keys = ["llm", "data_paths", "system"]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    # Validate LLM config
    if not config["llm"].get("api_key"):
        print("❌ Missing LLM API key")
        return False
    
    # Validate data paths
    for path_name, path in config["data_paths"].items():
        if not os.path.exists(path):
            print(f"❌ Data file not found: {path_name} -> {path}")
            return False
    
    return True