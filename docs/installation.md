# Installation Guide

This guide provides detailed instructions for installing Collab-Overcooked.

## Prerequisites

- Python 3.8 or higher
- Anaconda or Miniconda
- Git (for cloning the repository)

## Quick Installation

### Method 1: Using the Installation Script

1. Clone the repository:
```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked
```

2. Run the installation script:
```bash
bash scripts/install.sh
```

3. Activate the environment:
```bash
conda activate collab-overcooked
```

### Method 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate collab-overcooked
```

3. Install the main package (includes all dependencies):
```bash
pip install -e .
```

## Configuration

### OpenAI API Key

If you plan to use GPT models, you need to set up your OpenAI API key:

1. Create or edit the configuration file:
```bash
echo "your_openai_api_key_here" > configs/openai_key.txt
```

2. Make sure to keep this file private and don't commit it to version control.

### Custom Configuration

You can customize the default configuration by editing `configs/default.yaml`:

```yaml
# Environment settings
environment:
  horizon: 10
  order: "boiled_egg"
  layout: "cramped_room"
  
# Agent settings
agents:
  num_agents: 2
  agent_0:
    type: "gpt"
    model: "gpt-3.5-turbo"
    temperature: 0.1
```

## Verification

To verify that the installation was successful, run the quick test:

```bash
bash scripts/quick_test.sh
```

This will run a simple test scenario with a 3-step horizon and the boiled_egg task.

## Troubleshooting

### Common Issues

1. **Conda environment creation fails**: Make sure you have conda installed and updated.

2. **Permission errors**: Make sure the scripts have execute permissions:
   ```bash
   chmod +x scripts/*.sh
   ```

3. **OpenAI API errors**: Verify that your API key is correctly set in `configs/openai_key.txt`.

4. **Import errors**: Make sure you've activated the conda environment:
   ```bash
   conda activate collab-overcooked
   ```

### Getting Help

If you encounter issues not covered here, please:

1. Check the [Usage Guide](usage.md)
2. Look at the [API Reference](api_reference.md)
3. Open an issue on GitHub