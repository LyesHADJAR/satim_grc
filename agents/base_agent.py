"""
Enhanced Base Agent with performance tracking and collaboration capabilities
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone

class EnhancedBaseAgent(ABC):
    """Enhanced base class for all GRC agents with performance tracking."""
    
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        """
        Initialize the enhanced base agent.
        
        Args:
            name: Unique identifier for the agent
            llm_config: Configuration for the language model
        """
        self.name = name
        self.llm_config = llm_config
        self.logger = logging.getLogger(f"agent.{name}")
        self.logger.info(f"Initialized enhanced {name} agent")
        
        # Enhanced capabilities
        self.memory = []
        self.performance_metrics = {
            "requests_processed": 0,
            "successful_collaborations": 0,
            "avg_response_time": 0.0,
            "analysis_quality_score": 0.0
        }
        self.collaboration_history = []
        
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
        """Store data in the agent's memory with timestamp."""
        memory_entry = {
            "timestamp": datetime.now(timezone.utc),
            "data": data
        }
        self.memory.append(memory_entry)
        
        # Keep only recent memory (last 100 entries)
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def get_memory(self, limit: Optional[int] = None) -> List[Any]:
        """Retrieve data from the agent's memory."""
        if limit:
            return self.memory[-limit:]
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []
    
    def update_performance_metrics(self, **kwargs) -> None:
        """Update performance metrics."""
        for key, value in kwargs.items():
            if key in self.performance_metrics:
                self.performance_metrics[key] = value
    
    def update_quality_score(self, score: float) -> None:
        """Update analysis quality score with running average."""
        current_score = self.performance_metrics["analysis_quality_score"]
        if current_score == 0:
            self.performance_metrics["analysis_quality_score"] = score
        else:
            # Running average
            self.performance_metrics["analysis_quality_score"] = (current_score * 0.8 + score * 0.2)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "agent_name": self.name,
            "metrics": self.performance_metrics.copy(),
            "memory_size": len(self.memory),
            "collaboration_count": len(self.collaboration_history)
        }