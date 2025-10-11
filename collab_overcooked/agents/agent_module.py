"""
Refactored agent module with clean LLM integration
"""

import os
import time
import pandas as pd
import numpy as np
from scipy import spatial
from typing import Union, Dict, List, Optional, Any, Tuple
from rich import print as rprint

from .llm_client import LLMClientFactory, create_llm_config_from_yaml, LLMConfig
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff


class AgentModule:
    """
    Refactored agent module for LLM communication with clean configuration
    """
    
    def __init__(
        self,
        role_messages: List[Dict[str, str]],
        llm_config: Union[Dict[str, Any], LLMConfig],
        retrieval_method: str = "recent_k",
        k: int = 3,
        agent_name: str = "Agent"
    ):
        """
        Initialize agent module
        
        Args:
            role_messages: System/role messages for the agent
            llm_config: LLM configuration dict or LLMConfig object
            retrieval_method: Method for retrieving conversation history
            k: Number of recent messages to keep
            agent_name: Name of the agent (Chef/Assistant)
        """
        # Setup LLM client
        if isinstance(llm_config, dict):
            self.llm_config = create_llm_config_from_yaml(llm_config)
        else:
            self.llm_config = llm_config
            
        self.llm_client = LLMClientFactory.create_client(self.llm_config)
        
        # Agent configuration
        self.agent_name = agent_name
        self.retrieval_method = retrieval_method
        self.k = k
        
        # Message management
        self.role_messages = role_messages
        self.dialog_history = []
        self.current_user_message = None
        
        # Optional features
        self.embedding_cache = None
        self.experience_cache = []
        
    def add_system_message(self, message: Union[str, Dict[str, str]]):
        """Add a system message to role messages"""
        if isinstance(message, str):
            message = {"role": "system", "content": message}
        self.role_messages.append(message)
    
    def add_to_dialog_history(self, message: Dict[str, str]):
        """Add message to dialog history"""
        self.dialog_history.append(message)
        
        # Limit history size
        if len(self.dialog_history) > self.k * 4:  # Keep some buffer
            self.dialog_history = self.dialog_history[-self.k * 2:]
    
    def get_recent_context(self) -> List[Dict[str, str]]:
        """Get recent conversation context"""
        if self.retrieval_method == "recent_k":
            if self.k > 0:
                return self.dialog_history[-self.k:]
            else:
                return []
        else:
            return []
    
    def build_query_messages(self, user_input: str, include_context: bool = True) -> List[Dict[str, str]]:
        """Build complete message list for LLM query"""
        messages = []
        
        # Add system/role messages
        messages.extend(self.role_messages)
        
        # Add recent context if requested
        if include_context:
            context = self.get_recent_context()
            messages.extend(context)
        
        # Add current user input
        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        
        self.current_user_message = user_message
        return messages
    
    def query(
        self,
        user_input: str,
        include_context: bool = True,
        debug: bool = False
    ) -> Tuple[str, int]:
        """
        Query the LLM with user input
        
        Args:
            user_input: User input string
            include_context: Whether to include conversation history
            debug: Enable debug output
            
        Returns:
            tuple: (response_content, token_count)
        """
        try:
            # Build messages
            messages = self.build_query_messages(user_input, include_context)
            
            if debug:
                rprint(f"[blue][DEBUG][/blue] Querying {self.llm_config.model_name}")
                rprint(f"[blue][DEBUG][/blue] Messages: {len(messages)} total")
            
            # Get response from LLM
            response_content, token_count = self.llm_client.generate(messages)
            
            # Add to dialog history
            self.add_to_dialog_history(self.current_user_message)
            self.add_to_dialog_history({
                "role": "assistant", 
                "content": response_content
            })
            
            if debug:
                rprint(f"[green][SUCCESS][/green] Response: {len(response_content)} chars, {token_count} tokens")
            
            return response_content, token_count
            
        except Exception as e:
            rprint(f"[red][ERROR][/red] LLM query failed: {e}")
            return "", 0
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.dialog_history = []
        self.current_user_message = None
    
    def reset(self, other_agent=None):
        """Reset agent state for compatibility with overcooked framework"""
        self.reset_conversation()
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text"""
        return self.llm_client.count_tokens(text)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state"""
        return {
            "agent_name": self.agent_name,
            "model": self.llm_config.model_name,
            "model_type": self.llm_config.model_type,
            "dialog_length": len(self.dialog_history),
            "total_tokens": sum(
                self.get_token_count(msg.get("content", "")) 
                for msg in self.dialog_history
            )
        }


class CollaborativeAgent:
    """
    High-level collaborative agent that combines AgentModule with game logic
    """
    
    def __init__(
        self,
        agent_config: Dict[str, Any],
        role: str = "Chef",  # Chef or Assistant
        prompts_dir: str = "./collab_overcooked/prompts"
    ):
        """
        Initialize collaborative agent
        
        Args:
            agent_config: Agent configuration from YAML
            role: Agent role (Chef/Assistant)
            prompts_dir: Directory containing prompt templates
        """
        self.role = role
        self.prompts_dir = prompts_dir
        self.agent_index = None  # Will be set by AgentGroup
        
        # Load role-specific prompts
        self.role_messages = self._load_role_prompts()
        
        # Initialize agent module
        self.agent_module = AgentModule(
            role_messages=self.role_messages,
            llm_config=agent_config,
            retrieval_method=agent_config.get("retrieval_method", "recent_k"),
            k=agent_config.get("history_k", 3),
            agent_name=role
        )
        
        # Game-specific state
        self.current_timestep = None
        self.statistics = self._init_statistics()
    
    def set_agent_index(self, index: int):
        """Set agent index for compatibility with overcooked framework"""
        self.agent_index = index
    
    def reset(self, other_agent=None):
        """Reset agent state for new episode"""
        self.current_timestep = None
        self.statistics = self._init_statistics()
        # Reset the underlying agent module
        self.agent_module.reset(other_agent)
    
    def _load_role_prompts(self) -> List[Dict[str, str]]:
        """Load role-specific prompt templates"""
        # This would load from prompt files
        # For now, return basic system message
        return [{
            "role": "system",
            "content": f"You are a {self.role} in a collaborative cooking game. "
                      f"Work with your teammate to complete cooking tasks efficiently."
        }]
    
    def _init_statistics(self) -> Dict[str, Any]:
        """Initialize statistics tracking"""
        return {
            "total_queries": 0,
            "total_tokens": 0,
            "errors": [],
            "actions": [],
        }
    
    def get_action(
        self,
        observation: str,
        legal_actions: List[str],
        game_state: Optional[Dict] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get action from agent given current observation
        
        Args:
            observation: Current game observation
            legal_actions: List of legal actions
            game_state: Optional additional game state info
            
        Returns:
            tuple: (selected_action, metadata)
        """
        # Build input prompt
        input_prompt = self._build_action_prompt(observation, legal_actions, game_state)
        
        # Query LLM
        response, token_count = self.agent_module.query(input_prompt, debug=True)
        
        # Parse response to extract action
        action, metadata = self._parse_action_response(response, legal_actions)
        
        # Update statistics
        self.statistics["total_queries"] += 1
        self.statistics["total_tokens"] += token_count
        self.statistics["actions"].append({
            "action": action,
            "token_count": token_count,
            "timestamp": time.time()
        })
        
        return action, metadata
    
    def _build_action_prompt(
        self,
        observation: str,
        legal_actions: List[str],
        game_state: Optional[Dict] = None
    ) -> str:
        """Build prompt for action selection"""
        prompt = f"Current observation:\n{observation}\n\n"
        prompt += f"Legal actions: {', '.join(legal_actions)}\n\n"
        
        if game_state:
            prompt += f"Additional game state: {game_state}\n\n"
        
        prompt += f"As the {self.role}, what action do you want to take? "
        prompt += "Respond with your chosen action and brief reasoning."
        
        return prompt
    
    def _parse_action_response(
        self,
        response: str,
        legal_actions: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response to extract action"""
        # Simple parsing - look for legal actions in response
        response_lower = response.lower()
        
        for action in legal_actions:
            if action.lower() in response_lower:
                return action, {"reasoning": response}
        
        # Fallback to first legal action if parsing fails
        return legal_actions[0] if legal_actions else "stay", {
            "reasoning": response,
            "parse_error": "Could not extract valid action from response"
        }
    
    def reset(self, other_agent=None):
        """Reset agent state"""
        self.agent_module.reset_conversation()
        self.current_timestep = None
    
    def action(self, state):
        """Get action for overcooked framework compatibility"""
        # This is called by the overcooked framework
        # For now, return a simple action
        from overcooked_ai_py.mdp.actions import Action, Direction
        import random
        
        # Simple random action for testing
        directions = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST, Direction.STAY]
        random_direction = random.choice(directions)
        
        return Action(random_direction), None
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        base_stats = self.statistics.copy()
        base_stats.update(self.agent_module.get_conversation_summary())
        return base_stats


# Factory function for creating agents
def create_agent(
    agent_config: Dict[str, Any],
    role: str = "Chef",
    prompts_dir: str = "./collab_overcooked/prompts"
) -> CollaborativeAgent:
    """
    Factory function to create collaborative agents
    
    Args:
        agent_config: Agent configuration
        role: Agent role
        prompts_dir: Prompts directory
        
    Returns:
        CollaborativeAgent instance
    """
    return CollaborativeAgent(agent_config, role, prompts_dir)


# Backward compatibility - update statistics dicts
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
}

turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        "original_log": "",
    },
}