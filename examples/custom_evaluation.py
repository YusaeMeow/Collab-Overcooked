#!/usr/bin/env python
"""
Custom evaluation example for Collab-Overcooked

This example shows how to set up custom evaluation metrics
and run evaluation on your own task configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collab_overcooked.evaluation import evaluate_performance

def run_custom_evaluation():
    """Run custom evaluation with specific configurations."""
    
    print("Running custom evaluation example...")
    
    # Define custom evaluation configuration
    eval_config = {
        "tasks": ["boiled_egg", "soup"],
        "layouts": ["cramped_room", "coordination_ring"],
        "num_runs": 3,
        "metrics": [
            "f1_score",
            "similarity", 
            "redundancy",
            "collaboration_initiate",
            "collaboration_respond"
        ],
        "output_dir": "./custom_results"
    }
    
    agent_config = {
        "agent_0": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 512
        },
        "agent_1": {
            "model": "gpt-3.5-turbo", 
            "temperature": 0.2,
            "max_tokens": 512
        }
    }
    
    try:
        results = evaluate_performance(eval_config, agent_config)
        print("Custom evaluation completed!")
        print(f"Results saved to: {eval_config['output_dir']}")
        
        # Print summary
        print("\nEvaluation Summary:")
        for task in eval_config["tasks"]:
            if task in results:
                print(f"Task: {task}")
                for metric, value in results[task].items():
                    print(f"  {metric}: {value:.3f}")
                    
    except Exception as e:
        print(f"Error in custom evaluation: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    run_custom_evaluation()