from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

class BaseAgent(ABC):
    """Base class for all GRC agents."""
    
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for the agent
            llm_config: Configuration for the language model
        """
        self.name = name
        self.llm_config = llm_config
        self.logger = logging.getLogger(f"agent.{name}")
        self.logger.info(f"Initialized {name} agent")
        self.memory = []
    
    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Data to be processed by the agent
            
        Returns:
            Dict containing the processing results
        """
        pass
    
    def add_to_memory(self, data: Any) -> None:
        """Store data in the agent's memory."""
        self.memory.append(data)
        
    def get_memory(self) -> List[Any]:
        """Retrieve all data from the agent's memory."""
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []