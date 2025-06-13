from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import asyncio

class EnhancedBaseAgent(ABC):
    """Enhanced base class for all GRC agents with communication capabilities."""
    
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
        self.memory = []
        self.collaboration_history = []
        self.performance_metrics = {
            "requests_processed": 0,
            "successful_collaborations": 0,
            "analysis_quality_score": 0.0
        }
    
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
    
    # Collaboration methods for inter-agent communication
    async def analyze_content(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for other agents - to be implemented by subclasses."""
        self.logger.info(f"Content analysis request received: {request_data.get('domain', 'unknown')}")
        return {"status": "content_analysis_completed", "data": request_data}
    
    async def identify_gaps(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify gaps for other agents - to be implemented by subclasses."""
        self.logger.info(f"Gap identification request received: {request_data.get('domain', 'unknown')}")
        return {"status": "gap_identification_completed", "data": request_data}
    
    async def assess_coverage(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess coverage for other agents - to be implemented by subclasses."""
        self.logger.info(f"Coverage assessment request received: {request_data.get('domain', 'unknown')}")
        return {"status": "coverage_assessment_completed", "data": request_data}
    
    async def compare_policies(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare policies for other agents - to be implemented by subclasses."""
        self.logger.info(f"Policy comparison request received")
        return {"status": "policy_comparison_completed", "data": request_data}
    
    async def calculate_compliance_score(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance score for other agents - to be implemented by subclasses."""
        self.logger.info(f"Compliance scoring request received: {request_data.get('domain', 'unknown')}")
        return {"status": "compliance_scoring_completed", "data": request_data}
    
    async def process_generic_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process generic requests from other agents."""
        self.logger.info(f"Generic request received from agent")
        return {"status": "generic_request_processed", "data": request_data}
    
    def add_to_memory(self, data: Any) -> None:
        """Store data in the agent's memory."""
        self.memory.append({
            "timestamp": "2025-06-13 00:50:45",
            "data": data,
            "type": "memory_entry"
        })
        
        # Limit memory size
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]  # Keep last 50 entries
        
    def get_memory(self) -> List[Any]:
        """Retrieve all data from the agent's memory."""
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []
    
    def record_collaboration(self, collaboration_data: Dict[str, Any]) -> None:
        """Record collaboration activity."""
        self.collaboration_history.append({
            "timestamp": "2025-06-13 00:50:45",
            "collaboration_data": collaboration_data
        })
        
        # Update performance metrics
        self.performance_metrics["successful_collaborations"] += 1
        
        # Limit collaboration history
        if len(self.collaboration_history) > 50:
            self.collaboration_history = self.collaboration_history[-25:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            **self.performance_metrics,
            "memory_entries": len(self.memory),
            "collaboration_history_size": len(self.collaboration_history),
            "agent_name": self.name
        }
    
    def update_quality_score(self, new_score: float) -> None:
        """Update analysis quality score."""
        current_score = self.performance_metrics["analysis_quality_score"]
        # Rolling average
        self.performance_metrics["analysis_quality_score"] = (current_score * 0.7 + new_score * 0.3)