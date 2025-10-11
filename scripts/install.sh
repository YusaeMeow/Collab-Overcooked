#!/bin/bash
# Installation script for Collab-Overcooked

set -e

echo "Installing Collab-Overcooked..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate collab-overcooked

# Install the main package in development mode (includes all dependencies)
echo "Installing Collab-Overcooked in development mode..."
pip install -e .

echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "conda activate collab-overcooked"
echo ""
echo "To run a quick test, run:"
echo "bash scripts/quick_test.sh"