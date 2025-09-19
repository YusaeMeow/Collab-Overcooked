#!/bin/bash
# Evaluation script for Collab-Overcooked

set -e

echo "Running evaluation pipeline for Collab-Overcooked..."

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "collab-overcooked" ]]; then
    echo "Please activate the collab-overcooked environment first:"
    echo "conda activate collab-overcooked"
    exit 1
fi

cd collab_overcooked/evaluation

echo "Step 1: Running evaluation..."
python evaluation.py

echo "Step 2: Organizing results..."
python organize_result.py

echo "Step 3: Converting results..."
python convert_result.py

echo "Evaluation pipeline completed successfully!"
echo "Results can be found in the results directory."