"""
Collab-Overcooked: A Multi-Agent Collaborative Benchmark based on Overcooked-AI
"""

import os
import sys

# Add overcooked_ai_py to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
overcooked_ai_path = os.path.join(project_root, "dependencies", "overcooked_ai")
if overcooked_ai_path not in sys.path:
    sys.path.insert(0, overcooked_ai_path)

__version__ = "1.0.0"
__author__ = "Collab-Overcooked Team"

from .main import main

# Try to import optional components
try:
    from .agents.collab import LLMAgents
    from .agents.agent_module import CollaborativeAgent
    has_agents = True
except ImportError:
    LLMAgents = None
    CollaborativeAgent = None
    has_agents = False

try:
    from .environment.service import OvercookedService
    has_service = True
except ImportError:
    OvercookedService = None
    has_service = False

try:
    from .evaluation.evaluation import evaluate_performance
    has_evaluation = True
except ImportError:
    evaluate_performance = None
    has_evaluation = False

__all__ = ['main']

if has_agents:
    __all__.extend(['LLMAgents', 'CollaborativeAgent'])
if has_service:
    __all__.append('OvercookedService')
if has_evaluation:
    __all__.append('evaluate_performance')