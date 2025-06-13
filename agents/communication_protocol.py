"""
Agent Communication Protocol for multi-agent collaboration
"""
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
import uuid

class RequestType(Enum):
    """Types of requests between agents."""
    CONTENT_ANALYSIS = "content_analysis"
    GAP_IDENTIFICATION = "gap_identification"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_PLANNING = "strategy_planning"
    VALIDATION = "validation"

@dataclass
class AgentRequest:
    """Request structure for agent communication."""
    request_id: str
    requesting_agent: str
    target_agent: str
    request_type: RequestType
    data: Dict[str, Any]
    priority: int = 1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class AgentResponse:
    """Response structure for agent communication."""
    request_id: str
    responding_agent: str
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class AgentCommunicationProtocol:
    """Protocol for managing communication between agents."""
    
    def __init__(self):
        """Initialize the communication protocol."""
        self.logger = logging.getLogger("agent_communication")
        self.registered_agents: Dict[str, Any] = {}
        self.active_requests: Dict[str, AgentRequest] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.collaboration_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
    
    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent with the communication protocol."""
        self.registered_agents[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from the communication protocol."""
        if agent_name in self.registered_agents:
            del self.registered_agents[agent_name]
            self.logger.info(f"Unregistered agent: {agent_name}")
    
    async def request_analysis(self, request: AgentRequest) -> AgentResponse:
        """Send a request to another agent for analysis."""
        start_time = datetime.now(timezone.utc)
        
        # Validate request
        if request.target_agent not in self.registered_agents:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent="protocol",
                success=False,
                data={},
                error_message=f"Target agent {request.target_agent} not found"
            )
        
        # Store active request
        self.active_requests[request.request_id] = request
        self.collaboration_metrics["total_requests"] += 1
        
        try:
            # Get target agent
            target_agent = self.registered_agents[request.target_agent]
            
            # Process request based on type
            result_data = await self._process_request_by_type(target_agent, request)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create successful response
            response = AgentResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=True,
                data=result_data,
                processing_time=processing_time
            )
            
            # Update metrics
            self.collaboration_metrics["successful_requests"] += 1
            self._update_average_response_time(processing_time)
            
            # Log collaboration
            self._log_collaboration(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            response = AgentResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=False,
                data={},
                error_message=str(e),
                processing_time=processing_time
            )
            
            self.collaboration_metrics["failed_requests"] += 1
            return response
            
        finally:
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def _process_request_by_type(self, agent: Any, request: AgentRequest) -> Dict[str, Any]:
        """Process request based on type."""
        
        if request.request_type == RequestType.CONTENT_ANALYSIS:
            return await self._handle_content_analysis(agent, request)
        elif request.request_type == RequestType.GAP_IDENTIFICATION:
            return await self._handle_gap_identification(agent, request)
        elif request.request_type == RequestType.RISK_ASSESSMENT:
            return await self._handle_risk_assessment(agent, request)
        elif request.request_type == RequestType.VALIDATION:
            return await self._handle_validation(agent, request)
        else:
            # Default: pass data to agent's process method
            return await agent.process(request.data)
    
    async def _handle_content_analysis(self, agent: Any, request: AgentRequest) -> Dict[str, Any]:
        """Handle content analysis request."""
        domain = request.data.get("domain", "general")
        content = request.data.get("content", [])
        
        # Prepare analysis data
        analysis_data = {
            "domain": domain,
            "content_items": content,
            "analysis_type": "collaborative_content_analysis"
        }
        
        # Simulate content analysis (would call agent's specialized method)
        analyzed_content = []
        for item in content[:5]:  # Limit for demonstration
            analyzed_content.append({
                "content": item,
                "relevance_score": 0.8,
                "key_topics": [domain, "policy", "compliance"],
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return {
            "analyzed_content": analyzed_content,
            "domain": domain,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_gap_identification(self, agent: Any, request: AgentRequest) -> Dict[str, Any]:
        """Handle gap identification request."""
        # Placeholder for gap identification logic
        return {
            "gaps_identified": [],
            "gap_analysis_complete": True,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_risk_assessment(self, agent: Any, request: AgentRequest) -> Dict[str, Any]:
        """Handle risk assessment request."""
        # Placeholder for risk assessment logic
        return {
            "risk_level": "medium",
            "risk_factors": [],
            "assessment_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_validation(self, agent: Any, request: AgentRequest) -> Dict[str, Any]:
        """Handle validation request."""
        # Placeholder for validation logic
        return {
            "validation_status": "passed",
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _update_average_response_time(self, new_time: float) -> None:
        """Update average response time."""
        current_avg = self.collaboration_metrics["average_response_time"]
        total_requests = self.collaboration_metrics["total_requests"]
        
        if total_requests == 1:
            self.collaboration_metrics["average_response_time"] = new_time
        else:
            self.collaboration_metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + new_time) / total_requests
            )
    
    def _log_collaboration(self, request: AgentRequest, response: AgentResponse) -> None:
        """Log collaboration for history and metrics."""
        collaboration_log = {
            "request_id": request.request_id,
            "requesting_agent": request.requesting_agent,
            "target_agent": request.target_agent,
            "request_type": request.request_type.value,
            "success": response.success,
            "processing_time": response.processing_time,
            "timestamp": request.timestamp.isoformat()
        }
        
        self.request_history.append(collaboration_log)
        
        # Keep only recent history (last 1000 entries)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get collaboration metrics."""
        success_rate = 0.0
        if self.collaboration_metrics["total_requests"] > 0:
            success_rate = (self.collaboration_metrics["successful_requests"] / 
                          self.collaboration_metrics["total_requests"]) * 100
        
        return {
            **self.collaboration_metrics,
            "success_rate": success_rate,
            "active_requests": len(self.active_requests),
            "registered_agents": list(self.registered_agents.keys()),
            "history_size": len(self.request_history)
        }
    
    def get_agent_collaboration_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get collaboration statistics for a specific agent."""
        agent_requests = [
            log for log in self.request_history 
            if log["requesting_agent"] == agent_name or log["target_agent"] == agent_name
        ]
        
        requests_sent = len([log for log in agent_requests if log["requesting_agent"] == agent_name])
        requests_received = len([log for log in agent_requests if log["target_agent"] == agent_name])
        successful_collaborations = len([log for log in agent_requests if log["success"]])
        
        avg_response_time = 0.0
        if agent_requests:
            avg_response_time = sum(log["processing_time"] for log in agent_requests) / len(agent_requests)
        
        return {
            "agent_name": agent_name,
            "requests_sent": requests_sent,
            "requests_received": requests_received,
            "successful_collaborations": successful_collaborations,
            "average_response_time": avg_response_time,
            "collaboration_success_rate": (successful_collaborations / len(agent_requests) * 100) if agent_requests else 0
        }