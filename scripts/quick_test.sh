#!/bin/bash
# Quick test script for Collab-Overcooked

set -e

echo "Running quick test for Collab-Overcooked..."

# Function to check YAML configuration
check_yaml_config() {
    local config_file=""
    
    # Check for personal config first (highest priority)
    if [ -f "configs/test_personal.yaml" ]; then
        config_file="configs/test_personal.yaml"
        echo "✅ Using personal config: $config_file"
    # Check for default config (second priority)
    elif [ -f "configs/default.yaml" ]; then
        config_file="configs/default.yaml"
        echo "✅ Using default config: $config_file"
    # Check for any other YAML config files
    elif ls configs/*.yaml >/dev/null 2>&1; then
        config_file=$(ls configs/*.yaml | head -n1)
        echo "✅ Using config file: $config_file"
    else
        echo "❌ No YAML configuration found!"
        echo ""
        echo "Please create a YAML config file with API key:"
        echo "You can:"
        echo "1. Copy configs/default.yaml to configs/test_personal.yaml and update API keys"
        echo "2. Or modify configs/default.yaml directly"
        echo ""
        echo "Example config structure:"
        echo "agents:"
        echo "  agent_0:"
        echo "    type: \"openai\"       # or \"custom_api\" or \"vllm\""
        echo "    model: \"gpt-3.5-turbo\""
        echo "    api_key: \"your_api_key_here\""
        echo "    base_url: \"https://api.openai.com/v1\"  # for custom API"
        echo ""
        exit 1
    fi
    
    echo "Config file: $config_file"
    return 0
}

# Check YAML configuration
check_yaml_config

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "collab-overcooked" ]]; then
    echo "Please activate the collab-overcooked environment first:"
    echo "conda activate collab-overcooked"
    exit 1
fi

# Determine which config to use and run
if [ -f "configs/test_personal.yaml" ]; then
    echo "Running test with personal YAML config..."
    python -c "from collab_overcooked.main import main; main(config_path='configs/test_personal.yaml')"
elif [ -f "configs/default.yaml" ]; then
    echo "Running test with default YAML config..."
    python -c "from collab_overcooked.main import main; main(config_path='configs/default.yaml')"
elif ls configs/*.yaml >/dev/null 2>&1; then
    config_file=$(ls configs/*.yaml | head -n1)
    echo "Running test with config file: $config_file"
    python -c "from collab_overcooked.main import main; main(config_path='$config_file')"
else
    echo "No valid config found, exiting..."
    exit 1
fi

echo "Quick test completed successfully!"