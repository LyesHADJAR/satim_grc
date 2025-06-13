from typing import Dict, Any, List, Optional
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

class RequestType(Enum):
    CONTENT_ANALYSIS = "content_analysis"
    GAP_IDENTIFICATION = "gap_identification"
    COVERAGE_ASSESSMENT = "coverage_assessment"
    POLICY_COMPARISON = "policy_comparison"
    COMPLIANCE_SCORING = "compliance_scoring"

@dataclass
class AgentRequest:
    """Request structure for inter-agent communication."""
    request_id: str
    requesting_agent: str
    target_agent: str
    request_type: RequestType
    data: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0

@dataclass
class AgentResponse:
    """Response structure for inter-agent communication."""
    request_id: str
    responding_agent: str
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0

class AgentCommunicationProtocol:
    """Handles communication between agents for collaborative analysis."""
    
    def __init__(self):
        self.registered_agents = {}
        self.pending_requests = {}
        self.logger = logging.getLogger("agent_communication")
    
    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent for communication."""
        self.registered_agents[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")
    
    async def request_analysis(self, request: AgentRequest) -> AgentResponse:
        """Send analysis request to target agent."""
        if request.target_agent not in self.registered_agents:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=False,
                error_message=f"Agent {request.target_agent} not registered"
            )
        
        try:
            target_agent = self.registered_agents[request.target_agent]
            
            # Route request based on type
            if request.request_type == RequestType.CONTENT_ANALYSIS:
                result = await target_agent.analyze_content(request.data)
            elif request.request_type == RequestType.GAP_IDENTIFICATION:
                result = await target_agent.identify_gaps(request.data)
            elif request.request_type == RequestType.COVERAGE_ASSESSMENT:
                result = await target_agent.assess_coverage(request.data)
            elif request.request_type == RequestType.POLICY_COMPARISON:
                result = await target_agent.compare_policies(request.data)
            elif request.request_type == RequestType.COMPLIANCE_SCORING:
                result = await target_agent.calculate_compliance_score(request.data)
            else:
                result = await target_agent.process_generic_request(request.data)
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=True,
                data=result
            )
            
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=False,
                error_message=str(e)
            )
    
    async def broadcast_request(self, request_type: RequestType, data: Dict[str, Any], 
                               exclude_agents: List[str] = None) -> Dict[str, AgentResponse]:
        """Broadcast request to all registered agents."""
        exclude_agents = exclude_agents or []
        responses = {}
        
        tasks = []
        for agent_name in self.registered_agents:
            if agent_name not in exclude_agents:
                request = AgentRequest(
                    request_id=f"broadcast_{len(tasks)}",
                    requesting_agent="coordinator",
                    target_agent=agent_name,
                    request_type=request_type,
                    data=data
                )
                tasks.append(self.request_analysis(request))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            agent_name = list(self.registered_agents.keys())[i]
            if not isinstance(result, Exception):
                responses[agent_name] = result
            else:
                responses[agent_name] = AgentResponse(
                    request_id=f"broadcast_{i}",
                    responding_agent=agent_name,
                    success=False,
                    error_message=str(result)
                )
        
        return responses