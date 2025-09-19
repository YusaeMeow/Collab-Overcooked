import numpy as np
import os

from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.agents.agent import GreedyHumanModel, StayAgent, RandomAgent
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_dict_from_file, load_pickle

from collab_overcooked.agents.collab import LLMAgents
from collections import defaultdict

# Keep backward compatibility with old modules
try:
    from collab_overcooked.agents.modules import EMBEDDING_MODEL
except ImportError:
    EMBEDDING_MODEL = "text-embedding-3-small"


def make_agent(alg: str, mdp, layout, **gptargs):
    """
    Create an agent based on algorithm type
    Updated to support new LLM configuration system
    """
    
    if alg == "Stay":
        agent = StayAgent()

    elif alg == "Random":
        agent = RandomAgent()

    elif alg == "LLMPair" or alg == "Greedy":
        MLAM_PARAMS = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }
        counter_locations = mdp.get_counter_locations()
        MLAM_PARAMS["counter_goals"] = counter_locations
        MLAM_PARAMS["counter_drop"] = counter_locations
        MLAM_PARAMS["counter_pickup"] = counter_locations

        if alg == "LLMPair":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            ).ml_action_manager
            agent = LLMAgents(mlam, layout, **gptargs)

        elif alg == "Greedy":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            )
            agent = GreedyHumanModel(mlam)

    else:
        raise ValueError("Unsupported algorithm.")

    agent.set_mdp(mdp)
    return agent


def make_agent_from_config(agent_config: dict, mdp, layout):
    """
    Create an agent from new-style configuration
    
    Args:
        agent_config: Agent configuration dict from YAML
        mdp: Overcooked MDP
        layout: Layout name
    
    Returns:
        Agent instance
    """
    from collab_overcooked.agents import create_agent
    
    # Convert config format
    llm_config = {
        "type": agent_config.get("type", "openai"),
        "model": agent_config.get("model", "gpt-3.5-turbo"),
        "temperature": agent_config.get("temperature", 0.1),
        "max_tokens": agent_config.get("max_tokens", 512),
        "base_url": agent_config.get("base_url"),
        "api_key": agent_config.get("api_key"),
        "timeout": agent_config.get("timeout", 30),
        "max_retries": agent_config.get("max_retries", 3)
    }
    
    role = agent_config.get("role", "Agent")
    
    # Create collaborative agent
    collab_agent = create_agent(llm_config, role=role)
    
    # Wrap in LLMAgents for compatibility
    MLAM_PARAMS = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
    counter_locations = mdp.get_counter_locations()
    MLAM_PARAMS["counter_goals"] = counter_locations
    MLAM_PARAMS["counter_drop"] = counter_locations
    MLAM_PARAMS["counter_pickup"] = counter_locations
    
    mlam = MediumLevelPlanner.from_pickle_or_compute(
        mdp, MLAM_PARAMS, force_compute=True
    ).ml_action_manager
    
    # Create wrapper agent that uses new LLM system
    agent = LLMAgentsWrapper(mlam, layout, collab_agent, role.lower())
    agent.set_mdp(mdp)
    
    return agent


class LLMAgentsWrapper:
    """
    Wrapper to make new CollaborativeAgent compatible with existing LLMAgents interface
    """
    
    def __init__(self, mlam, layout, collab_agent, actor):
        self.mlam = mlam
        self.layout = layout
        self.collab_agent = collab_agent
        self.actor = actor
        self.mdp = None
        
        # Compatibility attributes
        self.teammate_ml_actions = []
        self.turn_statistics_dict = {
            "timestamp": 0,
            "order_list": [],
            "actions": [],
            "map": "",
            "statistical_data": {
                "score": 0,
                "communication": {"call": 0, "turn": [], "token": []},
                "error": {
                    "format_error": {"error_num": 0, "error_message": []},
                    "validator_error": {"error_num": 0, "error_message": []},
                },
                "error_correction": {
                    "format_correction": {"correction_num": 0, "correction_tokens": []},
                    "validator_correction": {
                        "correction_num": 0,
                        "reflection_obtain": [],
                        "correction_tokens": [],
                    },
                },
            },
            "content": {
                "observation": [],
                "reflection": [],
                "content": [],
                "action_list": [],
                "original_log": "",
            },
        }
    
    def set_mdp(self, mdp):
        self.mdp = mdp
    
    def action(self, state):
        """Get action using new LLM system"""
        # Convert state to observation string
        observation = self.mdp.state_string(state)
        
        # Get legal actions
        legal_actions = [Action.to_char(action) for action in Action.ALL_ACTIONS]
        
        # Get action from collaborative agent
        action_str, metadata = self.collab_agent.get_action(
            observation, legal_actions, {"state": state}
        )
        
        # Convert back to Action
        try:
            action = Action.from_char(action_str)
        except:
            action = Action.STAY  # Fallback
        
        # Update statistics
        self.turn_statistics_dict["statistical_data"]["communication"]["call"] += 1
        self.turn_statistics_dict["statistical_data"]["communication"]["token"].append(
            metadata.get("token_count", 0)
        )
        
        return action, []  # Return action and empty ingredient list
    
    def reset(self):
        """Reset agent state"""
        self.collab_agent.reset()
        self.teammate_ml_actions = []


# make the example into embedding for retrieval
def get_example_embedding(example_path, save_path=""):
    """Get embeddings for examples - updated for new system"""
    input = ""
    import openai
    import os
    import pandas as pd

    key = ""
    del_index = []
    cwd = os.getcwd()
    
    # Try different key file locations
    key_files = [
        "configs/openai_key.txt",
        "configs/personal_api_key.txt",
        "openai_key.txt"  # legacy location
    ]
    
    for key_file in key_files:
        key_path = os.path.join(cwd, key_file)
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                key = f.read().strip()
            break
    
    if not key:
        raise ValueError("No API key found. Please set up configs/openai_key.txt or configs/personal_api_key.txt")
    
    # Use new OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=key)

    with open(example_path, "r") as f:
        input = f.read()
        if input[0] == "\n":
            input = input[1:]
        input = input.split("</example>")
        for index, l in enumerate(input):
            input[index] = input[index].strip("\n\n")
            input[index] = input[index].strip("<example>")
            if input[index] == "":
                del_index.append(index)

    for index in sorted(del_index, reverse=True):
        del input[index]
    
    BATCH_SIZE = 10  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(input), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(input))
        batch = input[batch_start:batch_end]
        # Only use the content before "[OUTPUT]" for embedding
        batch = list(
            map(lambda x: x[: x.index("[OUTPUT]")] if "[OUTPUT]" in x else x, batch)
        )
        print(f"Batch {batch_start} to {batch_end-1}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            embeddings.append(be.embedding)

    # Save embeddings
    df = pd.DataFrame({"text": input, "embedding": embeddings})
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


def combine_statistic_dict(turn_statistics_dict_agent0, turn_statistics_dict_agent1, map, reward):
    """Combine statistics from both agents"""
    # Implementation from original codebase
    combined = {
        "timestamp": turn_statistics_dict_agent0.get("timestamp", 0),
        "order_list": turn_statistics_dict_agent0.get("order_list", []),
        "actions": [
            turn_statistics_dict_agent0.get("actions", []),
            turn_statistics_dict_agent1.get("actions", [])
        ],
        "map": map,
        "statistical_data": {
            "score": reward,
            "communication": [
                turn_statistics_dict_agent0.get("statistical_data", {}).get("communication", {}),
                turn_statistics_dict_agent1.get("statistical_data", {}).get("communication", {})
            ],
            "error": [
                turn_statistics_dict_agent0.get("statistical_data", {}).get("error", {}),
                turn_statistics_dict_agent1.get("statistical_data", {}).get("error", {})
            ],
            "error_correction": [
                turn_statistics_dict_agent0.get("statistical_data", {}).get("error_correction", {}),
                turn_statistics_dict_agent1.get("statistical_data", {}).get("error_correction", {})
            ],
        },
        "content": {
            "observation": [
                turn_statistics_dict_agent0.get("content", {}).get("observation", []),
                turn_statistics_dict_agent1.get("content", {}).get("observation", [])
            ],
            "reflection": [
                turn_statistics_dict_agent0.get("content", {}).get("reflection", []),
                turn_statistics_dict_agent1.get("content", {}).get("reflection", [])
            ],
            "content": [
                turn_statistics_dict_agent0.get("content", {}).get("content", []),
                turn_statistics_dict_agent1.get("content", {}).get("content", [])
            ],
            "action_list": [
                turn_statistics_dict_agent0.get("content", {}).get("action_list", []),
                turn_statistics_dict_agent1.get("content", {}).get("action_list", [])
            ],
            "original_log": "",
        },
    }
    
    return combined