# Usage Guide

This guide explains how to use Collab-Overcooked for various tasks.

## Basic Usage

### Running a Simple Experiment

The simplest way to run an experiment is using the command-line interface:

```bash
collab-overcooked --horizon 10 --order boiled_egg --layout cramped_room
```

Or using the Python module directly:

```bash
cd collab_overcooked
python main.py --horizon 10 --order boiled_egg
```

### Available Tasks

Collab-Overcooked supports several cooking tasks:

- `boiled_egg`: Cook a boiled egg
- `soup`: Prepare soup
- `salad`: Make a salad

### Available Layouts

Different kitchen layouts are available:

- `cramped_room`: A small kitchen with limited space
- `coordination_ring`: A circular layout requiring coordination
- `asymmetric_advantages`: Layout with different advantages for each agent

## Configuration

### Using Configuration Files

You can specify configurations using YAML files:

```yaml
# config.yaml
environment:
  horizon: 20
  order: "soup"
  layout: "coordination_ring"

agents:
  agent_0:
    model: "gpt-4"
    temperature: 0.0
  agent_1:
    model: "gpt-3.5-turbo"
    temperature: 0.1
```

Then run:

```bash
collab-overcooked --config config.yaml
```

### Agent Configuration

You can configure different types of agents:

#### GPT Agents
```yaml
agents:
  agent_0:
    type: "gpt"
    model: "gpt-3.5-turbo"  # or gpt-4
    temperature: 0.1
    max_tokens: 512
```

#### Local LLM Agents
```yaml
agents:
  agent_0:
    type: "local_llm"
    model_path: "/path/to/your/model"
    temperature: 0.1
```

## Evaluation

### Running Evaluation

To run a comprehensive evaluation:

```bash
bash scripts/run_evaluation.sh
```

Or step by step:

```bash
cd collab_overcooked/evaluation

# Step 1: Run evaluation
python evaluation.py

# Step 2: Organize results
python organize_result.py

# Step 3: Convert results
python convert_result.py
```

### Custom Evaluation

You can create custom evaluation scripts:

```python
from collab_overcooked.evaluation import evaluate_performance

eval_config = {
    "tasks": ["boiled_egg", "soup"],
    "layouts": ["cramped_room"],
    "num_runs": 5,
    "metrics": ["f1_score", "similarity", "collaboration_initiate"]
}

results = evaluate_performance(eval_config)
```

### Evaluation Metrics

Collab-Overcooked provides several evaluation metrics:

- **F1 Score**: Measures action accuracy
- **Similarity**: Compares actions to reference templates
- **Redundancy**: Measures unnecessary actions
- **Collaboration Initiate**: Ability to start collaboration
- **Collaboration Respond**: Ability to respond to collaboration

## Advanced Usage

### Custom Agents

You can implement custom agents by inheriting from the base agent class:

```python
from collab_overcooked.agents import BaseAgent

class CustomAgent(BaseAgent):
    def get_action(self, state, legal_actions):
        # Your custom logic here
        return selected_action
    
    def communicate(self, message):
        # Your communication logic here
        return response
```

### Custom Environments

To add new tasks or modify existing ones:

1. Create new layout files in `dependencies/overcooked_ai/overcooked_ai_py/data/layouts/`
2. Update the environment configuration
3. Modify the evaluation scripts if needed

### Logging and Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or configure in your YAML file:

```yaml
logging:
  level: "DEBUG"
  save_logs: true
  log_dir: "./logs"
```

## Examples

### Basic Example

```python
from collab_overcooked import main

# Run with default settings
main()
```

### Custom Configuration Example

```python
from collab_overcooked.main import run_experiment

config = {
    "environment": {
        "horizon": 15,
        "order": "soup",
        "layout": "cramped_room"
    },
    "agents": {
        "agent_0": {"model": "gpt-3.5-turbo"},
        "agent_1": {"model": "gpt-3.5-turbo"}
    }
}

results = run_experiment(config)
```

## Best Practices

1. **Start Small**: Begin with simple tasks (e.g., boiled_egg) and short horizons
2. **Monitor Costs**: GPT API calls can be expensive for long experiments
3. **Save Results**: Always save your experimental results for analysis
4. **Use Version Control**: Track your configuration changes
5. **Validate Results**: Use multiple runs to ensure reproducibility

## Troubleshooting

### Common Issues

- **Slow Performance**: Reduce horizon or use smaller models
- **API Rate Limits**: Add delays between API calls
- **Memory Issues**: Reduce batch sizes or use gradient checkpointing

### Performance Tips

- Use local models for rapid prototyping
- Cache results when possible
- Use multiple processes for parallel evaluation