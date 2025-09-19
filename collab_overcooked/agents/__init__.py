"""
Agent modules for Collab-Overcooked
"""

# New modular LLM system
from .llm_client import (
    LLMConfig, 
    LLMClientFactory, 
    create_llm_config_from_yaml,
    PRESET_CONFIGS
)
from .agent_module import (
    AgentModule, 
    CollaborativeAgent, 
    create_agent,
    statistics_dict,
    turn_statistics_dict
)

# Legacy modules for backward compatibility
from .modules import Module
from .collab import *
from .utils import *

__all__ = [
    # New system
    'LLMConfig',
    'LLMClientFactory', 
    'create_llm_config_from_yaml',
    'PRESET_CONFIGS',
    'AgentModule',
    'CollaborativeAgent',
    'create_agent',
    'statistics_dict',
    'turn_statistics_dict',
    # Legacy
    'Module',
    'collab',
    'utils'
]