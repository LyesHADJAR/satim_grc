import os
import json
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """Utility for managing configuration settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.logger = logging.getLogger("config")
        self.config = {}
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Load configuration from environment variables
        self._load_from_env()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # LLM configuration
        if os.getenv("LLM_API_KEY"):
            self.config["llm"] = self.config.get("llm", {})
            self.config["llm"]["api_key"] = os.getenv("LLM_API_KEY")
            self.config["llm"]["model"] = os.getenv("LLM_MODEL", "gpt-4")
            
        # Vector DB configuration
        if os.getenv("VECTOR_DB_HOST"):
            self.config["vector_db"] = self.config.get("vector_db", {})
            self.config["vector_db"]["host"] = os.getenv("VECTOR_DB_HOST")
            self.config["vector_db"]["port"] = os.getenv("VECTOR_DB_PORT", "8000")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value