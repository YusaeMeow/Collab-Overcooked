<div align="center">
  <h1> Collab-Overcooked </h1>
  <p><em>A Multi-Agent Collaborative Benchmark based on Overcooked-AI</em></p>
</div>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

We propose a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, **Collab-Overcooked**, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives:

1. **Multi-agent Framework**: Supports diverse tasks and objectives while encouraging collaboration through natural language communication
2. **Process-oriented Evaluation**: Introduces comprehensive metrics to assess fine-grained collaboration capabilities of different LLM agents

## 🎯 Key Features

- **Multiple Cooking Tasks**: Boiled egg, soup, salad, and more
- **Diverse Kitchen Layouts**: Various configurations requiring different collaboration strategies  
- **LLM Agent Support**: Works with GPT models, local LLMs, and custom agents
- **Comprehensive Evaluation**: F1 score, similarity, redundancy, and collaboration metrics
- **Easy Installation**: One-command setup with conda environment
- **Flexible Configuration**: YAML-based configuration system
- **Rich Documentation**: Complete guides and API reference

## 🚀 Quick Start

### Installation

#### Option 1: Automatic Installation (Recommended)

```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked
bash scripts/install.sh
```

#### Option 2: Manual Installation

```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked

# Create conda environment
conda env create -f environment.yml
conda activate collab-overcooked

# Install dependencies
cd dependencies/overcooked_ai
pip install -e .
cd ../..

# Install main package
pip install -e .
```

### Configuration

1. **Set up API configuration**:
   Copy and edit the configuration file:
   ```bash
   cp configs/default.yaml configs/test_personal.yaml
   # Edit configs/test_personal.yaml and add your API key
   ```

2. **Customize configuration** (optional):
   Edit `configs/test_personal.yaml` to modify settings and add your API key

### Quick Test

Run a simple test to verify installation:

```bash
bash scripts/quick_test.sh
```

Or manually:

```bash
conda activate collab-overcooked
collab-overcooked --horizon 3 --order boiled_egg
```

This runs a 3-step collaboration scenario between two GPT agents making a boiled egg.

## 📖 Documentation

- **[Installation Guide](docs/installation.md)**: Detailed installation instructions
- **[Usage Guide](docs/usage.md)**: Comprehensive usage examples and tutorials  
- **[API Reference](docs/api_reference.md)**: Complete API documentation

## 🔧 Usage Examples

### Basic Usage

```bash
# Run a simple experiment
collab-overcooked --horizon 10 --order soup --layout cramped_room

# Run evaluation pipeline
bash scripts/run_evaluation.sh
```

### Python API

```python
from collab_overcooked import main
from collab_overcooked.evaluation import evaluate_performance

# Run experiment programmatically
results = main()

# Custom evaluation
eval_config = {
    "tasks": ["boiled_egg", "soup"],
    "layouts": ["cramped_room"],
    "num_runs": 5,
    "metrics": ["f1_score", "collaboration_initiate"]
}

results = evaluate_performance(eval_config)
```

### Local LLM Support

Configure local LLMs using [vLLM](https://github.com/vllm-project/vllm):

```yaml
agents:
  agent_0:
    type: "local_llm"
    model_path: "/path/to/your/model"
    temperature: 0.1
```

## 📊 Evaluation

### Automated Evaluation

```bash
bash scripts/run_evaluation.sh
```

This runs the complete evaluation pipeline:

1. **evaluation.py**: Calculates metrics for each task
2. **organize_result.py**: Summarizes results into `statistics_data.csv`  
3. **convert_result.py**: Computes complexity-level metrics in `converted_data.csv`

### Key Metrics

- **F1 Score**: Action accuracy using TES function
- **Similarity**: Comparison with Reference Action Templates (RATs)
- **Redundancy**: Unnecessary action detection
- **Collaboration Initiate**: Ability to start collaboration
- **Collaboration Respond**: Ability to respond to collaboration

## 🛠️ Customization

### Adding New Tasks

1. Create layout files in `dependencies/overcooked_ai/overcooked_ai_py/data/layouts/`
2. Update configuration files
3. Modify evaluation scripts if needed

### Custom Agents

```python
from collab_overcooked.agents import BaseAgent

class CustomAgent(BaseAgent):
    def get_action(self, state, legal_actions):
        # Your custom logic
        return selected_action
```

### Environment Modification

The environment logic is in `dependencies/overcooked_ai/`. Modify:
- Layout files in `data/layouts/` for new recipes/ingredients
- Environment logic in `mdp/` for new interactive elements


## Reference
```bibtex
@inproceedings{zhang2024proagent,
  title={Proagent: building proactive cooperative agents with large language models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}

@inproceedings{carroll2019utility,
 title={On the Utility of Learning About Humans for Human-AI Coordination},
 author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
 booktitle={Advances in Neural Information Processing Systems},
 pages={},
 volume={32},
 year={2019},
}
```

