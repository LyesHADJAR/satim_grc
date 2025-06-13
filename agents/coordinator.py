from typing import Dict, Any, List, Type, Optional
import asyncio
import logging
from .base_agent import EnhancedBaseAgent
from .communication_protocol import AgentCommunicationProtocol, AgentRequest, RequestType

class EnhancedAgentCoordinator:
    """
    Enhanced coordinator with sophisticated workflow management and agent collaboration.
    """
    
    def __init__(self):
        """Initialize the enhanced agent coordinator."""
        self.agents: Dict[str, EnhancedBaseAgent] = {}
        self.communication_protocol = AgentCommunicationProtocol()
        self.workflow_history = []
        self.logger = logging.getLogger("enhanced_coordinator")
        
    def register_agent(self, agent: EnhancedBaseAgent) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: The enhanced agent to register
        """
        self.agents[agent.name] = agent
        self.communication_protocol.register_agent(agent.name, agent)
        self.logger.info(f"Registered enhanced agent: {agent.name}")
    
    def get_agent(self, name: str) -> EnhancedBaseAgent:
        """
        Get an agent by name.
        
        Args:
            name: Name of the agent to retrieve
            
        Returns:
            The requested agent
            
        Raises:
            KeyError: If agent with the given name is not registered
        """
        if name not in self.agents:
            raise KeyError(f"No agent registered with name: {name}")
        return self.agents[name]
    
    async def execute_collaborative_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute collaborative analysis across multiple agents.
        
        Args:
            analysis_request: Analysis request containing domains, policies, etc.
            
        Returns:
            Comprehensive analysis results from agent collaboration
        """
        workflow_id = f"workflow_{len(self.workflow_history)}"
        self.logger.info(f"Starting collaborative analysis workflow: {workflow_id}")
        
        # Phase 1: Content Extraction and Preparation
        content_tasks = []
        domains = analysis_request.get("domains", [])
        
        for domain in domains:
            request = AgentRequest(
                request_id=f"{workflow_id}_content_{domain}",
                requesting_agent="coordinator",
                target_agent="enhanced_policy_analyzer",
                request_type=RequestType.CONTENT_ANALYSIS,
                data={
                    "domain": domain,
                    "company_policies": analysis_request.get("company_policy_ids", []),
                    "reference_policies": analysis_request.get("reference_policy_ids", [])
                }
            )
            content_tasks.append(self.communication_protocol.request_analysis(request))
        
        content_results = await asyncio.gather(*content_tasks)
        
        # Phase 2: Gap Analysis Collaboration
        gap_tasks = []
        for i, domain in enumerate(domains):
            if content_results[i].success:
                request = AgentRequest(
                    request_id=f"{workflow_id}_gaps_{domain}",
                    requesting_agent="coordinator",
                    target_agent="enhanced_policy_analyzer",
                    request_type=RequestType.GAP_IDENTIFICATION,
                    data={
                        "domain": domain,
                        "content_analysis": content_results[i].data
                    }
                )
                gap_tasks.append(self.communication_protocol.request_analysis(request))
        
        gap_results = await asyncio.gather(*gap_tasks)
        
        # Phase 3: Coverage Assessment Collaboration
        coverage_tasks = []
        for i, domain in enumerate(domains):
            if gap_results[i].success:
                request = AgentRequest(
                    request_id=f"{workflow_id}_coverage_{domain}",
                    requesting_agent="coordinator",
                    target_agent="enhanced_policy_analyzer",
                    request_type=RequestType.COVERAGE_ASSESSMENT,
                    data={
                        "domain": domain,
                        "gap_analysis": gap_results[i].data
                    }
                )
                coverage_tasks.append(self.communication_protocol.request_analysis(request))
        
        coverage_results = await asyncio.gather(*coverage_tasks)
        
        # Phase 4: Comprehensive Scoring
        scoring_tasks = []
        for i, domain in enumerate(domains):
            if coverage_results[i].success:
                request = AgentRequest(
                    request_id=f"{workflow_id}_scoring_{domain}",
                    requesting_agent="coordinator",
                    target_agent="enhanced_policy_analyzer",
                    request_type=RequestType.COMPLIANCE_SCORING,
                    data={
                        "domain": domain,
                        "analysis_data": {
                            "content": content_results[i].data if content_results[i].success else {},
                            "gaps": gap_results[i].data if gap_results[i].success else {},
                            "coverage": coverage_results[i].data if coverage_results[i].success else {}
                        }
                    }
                )
                scoring_tasks.append(self.communication_protocol.request_analysis(request))
        
        scoring_results = await asyncio.gather(*scoring_tasks)
        
        # Compile comprehensive results
        collaborative_results = {
            "workflow_id": workflow_id,
            "analysis_approach": "enhanced_collaborative_analysis",
            "domains_analyzed": domains,
            "domain_results": {},
            "collaboration_metrics": {
                "total_agent_requests": len(content_tasks) + len(gap_tasks) + len(coverage_tasks) + len(scoring_tasks),
                "successful_collaborations": sum(1 for result_set in [content_results, gap_results, coverage_results, scoring_results] 
                                               for result in result_set if result.success),
                "workflow_timestamp": "2025-06-13 00:50:45"
            }
        }
        
        # Process results for each domain
        for i, domain in enumerate(domains):
            domain_result = {
                "domain": domain,
                "content_analysis": content_results[i].data if i < len(content_results) and content_results[i].success else {},
                "gap_analysis": gap_results[i].data if i < len(gap_results) and gap_results[i].success else {},
                "coverage_analysis": coverage_results[i].data if i < len(coverage_results) and coverage_results[i].success else {},
                "compliance_scoring": scoring_results[i].data if i < len(scoring_results) and scoring_results[i].success else {},
                "collaboration_success": all([
                    i < len(content_results) and content_results[i].success,
                    i < len(gap_results) and gap_results[i].success,
                    i < len(coverage_results) and coverage_results[i].success,
                    i < len(scoring_results) and scoring_results[i].success
                ])
            }
            collaborative_results["domain_results"][domain] = domain_result
        
        # Record workflow
        self.workflow_history.append({
            "workflow_id": workflow_id,
            "timestamp": "2025-06-13 00:50:45",
            "domains": domains,
            "success": True,
            "metrics": collaborative_results["collaboration_metrics"]
        })
        
        return collaborative_results
    
    async def execute_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a multi-agent workflow with enhanced coordination.
        
        Args:
            workflow_steps: List of workflow steps with enhanced configuration
            
        Returns:
            Results from the workflow execution
        """
        workflow_id = f"enhanced_workflow_{len(self.workflow_history)}"
        result = {"workflow_id": workflow_id, "step_results": {}}
        
        for i, step in enumerate(workflow_steps):
            step_id = f"step_{i+1}"
            agent_name = step["agent_name"]
            self.logger.info(f"Workflow {workflow_id} - {step_id}: {agent_name}")
            
            # Get the agent
            agent = self.get_agent(agent_name)
            
            # Determine input for this step
            if "input_key" in step and step["input_key"] in result:
                input_data = result[step["input_key"]]
            elif "input_data" in step:
                input_data = step["input_data"]
            else:
                input_data = result.get("step_results", {})
                
            # Apply transform if specified
            if "transform" in step and callable(step["transform"]):
                input_data = step["transform"](input_data)
                
            # Execute step with error handling
            try:
                step_result = await agent.process(input_data)
                result["step_results"][step_id] = {
                    "agent": agent_name,
                    "success": True,
                    "result": step_result,
                    "timestamp": "2025-06-13 00:50:45"
                }
                
                # Update agent performance metrics
                agent.performance_metrics["requests_processed"] += 1
                
            except Exception as e:
                self.logger.error(f"Workflow step failed: {step_id} - {e}")
                result["step_results"][step_id] = {
                    "agent": agent_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": "2025-06-13 00:50:45"
                }
                
                # If step is critical, abort workflow
                if step.get("critical", False):
                    result["workflow_aborted"] = True
                    result["abort_reason"] = f"Critical step {step_id} failed: {e}"
                    break
            
            # Store result for next step
            if i == len(workflow_steps) - 1:
                result["final_result"] = result["step_results"][step_id].get("result", {})
                
        # Record workflow execution
        self.workflow_history.append({
            "workflow_id": workflow_id,
            "timestamp": "2025-06-13 00:50:45",
            "steps_count": len(workflow_steps),
            "success": not result.get("workflow_aborted", False),
            "performance": self._calculate_workflow_performance(result)
        })
                
        return result
    
    def _calculate_workflow_performance(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workflow performance metrics."""
        step_results = workflow_result.get("step_results", {})
        
        successful_steps = sum(1 for step in step_results.values() if step.get("success", False))
        total_steps = len(step_results)
        
        return {
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "workflow_aborted": workflow_result.get("workflow_aborted", False)
        }
    
    async def execute_parallel(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple agent tasks in parallel with enhanced error handling.
        
        Args:
            tasks: List of tasks with enhanced configuration
            
        Returns:
            List of results from all tasks
        """
        async def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
            try:
                agent = self.get_agent(task["agent_name"])
                result = await agent.process(task["input_data"])
                
                # Update performance metrics
                agent.performance_metrics["requests_processed"] += 1
                
                return {
                    "agent": task["agent_name"],
                    "success": True,
                    "result": result,
                    "timestamp": "2025-06-13 00:50:45"
                }
            except Exception as e:
                self.logger.error(f"Parallel task failed for {task['agent_name']}: {e}")
                return {
                    "agent": task["agent_name"],
                    "success": False,
                    "error": str(e),
                    "timestamp": "2025-06-13 00:50:45"
                }
        
        # Execute all tasks in parallel
        task_results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                processed_results.append({
                    "agent": tasks[i]["agent_name"] if i < len(tasks) else "unknown",
                    "success": False,
                    "error": str(result),
                    "timestamp": "2025-06-13 00:50:45"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        return self.workflow_history
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all registered agents."""
        summary = {
            "total_agents": len(self.agents),
            "agent_metrics": {},
            "overall_performance": {
                "total_requests": 0,
                "total_collaborations": 0,
                "average_quality_score": 0.0
            }
        }
        
        total_quality_score = 0
        for agent_name, agent in self.agents.items():
            metrics = agent.get_performance_metrics()
            summary["agent_metrics"][agent_name] = metrics
            summary["overall_performance"]["total_requests"] += metrics["requests_processed"]
            summary["overall_performance"]["total_collaborations"] += metrics["successful_collaborations"]
            total_quality_score += metrics["analysis_quality_score"]
        
        if len(self.agents) > 0:
            summary["overall_performance"]["average_quality_score"] = total_quality_score / len(self.agents)
        
        return summary