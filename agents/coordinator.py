from typing import Dict, Any, List, Type
import asyncio
import logging
from .base_agent import BaseAgent

class AgentCoordinator:
    """
    Coordinates the workflow between multiple specialized agents.
    """
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("coordinator")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> BaseAgent:
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
    
    async def execute_workflow(self, 
                              workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a multi-agent workflow.
        
        Args:
            workflow_steps: List of steps, each containing:
                - agent_name: Name of the agent to use
                - input_key: Key to extract from previous step's output (optional)
                - input_data: Static input data (optional)
                - transform: Function to transform data before passing to agent (optional)
                
        Returns:
            Results from the final step in the workflow
        """
        result = {}
        
        for i, step in enumerate(workflow_steps):
            agent_name = step["agent_name"]
            self.logger.info(f"Workflow step {i+1}: {agent_name}")
            
            # Get the agent
            agent = self.get_agent(agent_name)
            
            # Determine input for this step
            if "input_key" in step and step["input_key"] in result:
                input_data = result[step["input_key"]]
            elif "input_data" in step:
                input_data = step["input_data"]
            else:
                input_data = result  # Use entire previous result
                
            # Apply transform if specified
            if "transform" in step and callable(step["transform"]):
                input_data = step["transform"](input_data)
                
            # Process with the agent
            step_result = await agent.process(input_data)
            
            # Store the result
            result[agent_name] = step_result
            
            # If this is the last step, return its result directly
            if i == len(workflow_steps) - 1:
                return step_result
                
        return result

    async def execute_parallel(self, 
                              tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple agent tasks in parallel.
        
        Args:
            tasks: List of tasks, each containing:
                - agent_name: Name of the agent to use
                - input_data: Input data for the agent
                
        Returns:
            List of results from all tasks
        """
        async_tasks = []
        
        for task in tasks:
            agent = self.get_agent(task["agent_name"])
            async_tasks.append(agent.process(task["input_data"]))
            
        return await asyncio.gather(*async_tasks)