# API Reference

This document provides detailed information about the Collab-Overcooked API.

## Core Modules

### collab_overcooked.main

The main entry point for running experiments.

#### Functions

##### `main(args=None)`

Main function to run Collab-Overcooked experiments.

**Parameters:**
- `args` (Optional[Namespace]): Command line arguments or custom configuration

**Returns:**
- `dict`: Experiment results

**Example:**
```python
from collab_overcooked import main

# Run with default settings
results = main()

# Run with custom args
import argparse
args = argparse.Namespace(horizon=10, order='boiled_egg')
results = main(args)
```

### collab_overcooked.agents

Agent-related modules for multi-agent collaboration.

#### Classes

##### `CollabAgent`

Base class for collaborative agents.

**Methods:**

###### `__init__(self, config)`

Initialize the collaborative agent.

**Parameters:**
- `config` (dict): Agent configuration including model, temperature, etc.

###### `get_action(self, state, legal_actions)`

Get the next action for the agent.

**Parameters:**
- `state` (State): Current environment state
- `legal_actions` (List[Action]): Available legal actions

**Returns:**
- `Action`: Selected action

###### `communicate(self, message)`

Handle communication with other agents.

**Parameters:**
- `message` (str): Incoming message

**Returns:**
- `str`: Response message

### collab_overcooked.environment

Environment-related modules.

#### Classes

##### `OvercookedService`

Service class for managing the Overcooked environment.

**Methods:**

###### `__init__(self, config)`

Initialize the service.

**Parameters:**
- `config` (dict): Environment configuration

###### `reset(self)`

Reset the environment to initial state.

**Returns:**
- `State`: Initial state

###### `step(self, actions)`

Execute actions in the environment.

**Parameters:**
- `actions` (List[Action]): Actions for each agent

**Returns:**
- `Tuple[State, List[float], bool, dict]`: Next state, rewards, done flag, info

### collab_overcooked.evaluation

Evaluation and metrics modules.

#### Functions

##### `evaluate_performance(eval_config, agent_config=None)`

Evaluate agent performance on specified tasks.

**Parameters:**
- `eval_config` (dict): Evaluation configuration
- `agent_config` (Optional[dict]): Agent configuration

**Returns:**
- `dict`: Evaluation results

**Example:**
```python
from collab_overcooked.evaluation import evaluate_performance

eval_config = {
    "tasks": ["boiled_egg", "soup"],
    "layouts": ["cramped_room"],
    "num_runs": 3,
    "metrics": ["f1_score", "similarity"]
}

results = evaluate_performance(eval_config)
```

##### `calculate_f1_score(predicted_actions, reference_actions)`

Calculate F1 score between predicted and reference actions.

**Parameters:**
- `predicted_actions` (List[Action]): Predicted actions
- `reference_actions` (List[Action]): Reference actions

**Returns:**
- `float`: F1 score

##### `calculate_similarity(actions_a, actions_b)`

Calculate similarity between two action sequences.

**Parameters:**
- `actions_a` (List[Action]): First action sequence
- `actions_b` (List[Action]): Second action sequence

**Returns:**
- `float`: Similarity score

## Configuration

### Environment Configuration

```python
env_config = {
    "horizon": 10,          # Number of time steps
    "order": "boiled_egg",  # Task type
    "layout": "cramped_room", # Kitchen layout
    "max_steps": 400        # Maximum environment steps
}
```

### Agent Configuration

```python
agent_config = {
    "type": "gpt",           # Agent type
    "model": "gpt-3.5-turbo", # Model name
    "temperature": 0.1,      # Sampling temperature
    "max_tokens": 512        # Maximum tokens per response
}
```

### Evaluation Configuration

```python
eval_config = {
    "metrics": [             # Metrics to calculate
        "f1_score",
        "similarity", 
        "redundancy",
        "collaboration_initiate",
        "collaboration_respond"
    ],
    "output_dir": "./results", # Output directory
    "save_trajectories": True  # Save full trajectories
}
```

## Data Types

### Action

Represents an action in the environment.

**Attributes:**
- `action_type` (str): Type of action (e.g., "move", "interact")
- `direction` (Optional[Tuple[int, int]]): Movement direction
- `target` (Optional[str]): Interaction target

### State

Represents the environment state.

**Attributes:**
- `players` (List[Player]): Player positions and states
- `objects` (Dict[str, Object]): Environment objects
- `time_left` (int): Remaining time steps

### Player

Represents a player in the environment.

**Attributes:**
- `position` (Tuple[int, int]): Player position
- `orientation` (Tuple[int, int]): Player orientation
- `held_object` (Optional[Object]): Object being held

## Exceptions

### `CollabOvercookedError`

Base exception class for Collab-Overcooked errors.

### `ConfigurationError`

Raised when there are configuration issues.

### `EnvironmentError`

Raised when there are environment-related errors.

### `AgentError`

Raised when there are agent-related errors.

## Utilities

### collab_overcooked.utils

Utility functions for common operations.

#### Functions

##### `load_config(config_path)`

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `dict`: Configuration dictionary

##### `save_results(results, output_path)`

Save results to file.

**Parameters:**
- `results` (dict): Results to save
- `output_path` (str): Output file path

##### `setup_logging(level="INFO", log_file=None)`

Setup logging configuration.

**Parameters:**
- `level` (str): Logging level
- `log_file` (Optional[str]): Log file path

## Examples

### Basic Usage

```python
import collab_overcooked

# Simple experiment
results = collab_overcooked.main()

# Custom configuration
from collab_overcooked import main
import argparse

args = argparse.Namespace(
    horizon=15,
    order="soup",
    layout="coordination_ring"
)

results = main(args)
```

### Advanced Usage

```python
from collab_overcooked.agents import CollabAgent
from collab_overcooked.environment import OvercookedService
from collab_overcooked.evaluation import evaluate_performance

# Custom agent setup
agent_config = {
    "type": "gpt",
    "model": "gpt-4",
    "temperature": 0.0
}

agent = CollabAgent(agent_config)

# Custom environment setup
env_config = {
    "layout": "cramped_room",
    "horizon": 20
}

env = OvercookedService(env_config)

# Custom evaluation
eval_config = {
    "tasks": ["boiled_egg"],
    "num_runs": 5,
    "metrics": ["f1_score", "collaboration_initiate"]
}

results = evaluate_performance(eval_config)
```