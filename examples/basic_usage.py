#!/usr/bin/env python
"""
Basic usage example for Collab-Overcooked

This example demonstrates how to run a simple collaborative cooking task
with two GPT agents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collab_overcooked import main

def run_basic_example():
    """Run a basic collaborative cooking example."""
    
    print("Running basic Collab-Overcooked example...")
    print("Task: Two agents collaborating to make boiled egg")
    print("Horizon: 5 steps")
    print()
    
    # Set up arguments for the main function
    import argparse
    
    # Create a mock args object
    class Args:
        horizon = 5
        order = "boiled_egg"
        layout = "cramped_room"
        agent_0_model = "gpt-3.5-turbo"
        agent_1_model = "gpt-3.5-turbo"
        temperature = 0.1
        max_tokens = 512
        save_results = True
        
    args = Args()
    
    # Run the main function
    try:
        main(args)
        print("Example completed successfully!")
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have:")
        print("1. Activated the collab-overcooked conda environment")
        print("2. Set up your OpenAI API key in configs/openai_key.txt")
        print("3. Installed all dependencies")

if __name__ == "__main__":
    run_basic_example()