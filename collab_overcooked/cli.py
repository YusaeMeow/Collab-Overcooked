#!/usr/bin/env python
"""
Command line interface for Collab-Overcooked
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from collab_overcooked.main import main, load_config_from_yaml


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Collab-Overcooked: Multi-Agent Collaborative Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with YAML config (recommended)
  collab-overcooked --config configs/test_personal.yaml
  
  # Run with command line args (legacy)
  collab-overcooked --horizon 5 --order boiled_egg --model gpt-3.5-turbo
  
  # Quick test
  collab-overcooked --config configs/default.yaml --horizon 3
        """
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Legacy command line arguments
    parser.add_argument(
        '--horizon',
        type=int,
        default=10,
        help='Number of time steps to run (default: 10)'
    )
    
    parser.add_argument(
        '--order',
        type=str,
        default='boiled_egg',
        choices=['boiled_egg', 'soup', 'salad'],
        help='Task to perform (default: boiled_egg)'
    )
    
    parser.add_argument(
        '--layout',
        type=str,
        default='cramped_room',
        help='Kitchen layout (default: cramped_room)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='LLM model to use (default: gpt-3.5-turbo)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for LLM generation (default: 0.1)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens for LLM generation (default: 512)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        help='Base URL for custom API (e.g., https://api2.aigcbest.top)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (will use config file if not provided)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    return parser


def args_to_config(args):
    """Convert command line args to configuration dict"""
    
    # Determine model type based on base_url
    if args.base_url:
        model_type = "custom_api"
    else:
        model_type = "openai"
    
    config = {
        'environment': {
            'horizon': args.horizon,
            'order': args.order,
            'layout': args.layout,
            'max_steps': 400
        },
        'agents': {
            'num_agents': 2,
            'agent_0': {
                'type': model_type,
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'role': 'Chef'
            },
            'agent_1': {
                'type': model_type,
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'role': 'Assistant'
            }
        }
    }
    
    # Add optional parameters
    if args.base_url:
        config['agents']['agent_0']['base_url'] = args.base_url
        config['agents']['agent_1']['base_url'] = args.base_url
    
    if args.api_key:
        config['agents']['agent_0']['api_key'] = args.api_key
        config['agents']['agent_1']['api_key'] = args.api_key
    
    return config


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        if args.config:
            # Load configuration from YAML file
            print(f"Loading configuration from: {args.config}")
            if not os.path.exists(args.config):
                print(f"Error: Configuration file not found: {args.config}")
                sys.exit(1)
            
            main(config_path=args.config)
            
        else:
            # Use command line arguments
            print("Using command line configuration")
            config = args_to_config(args)
            
            # Convert to variant format
            from collab_overcooked.main import convert_yaml_to_variant
            variant = convert_yaml_to_variant(config)
            variant['yaml_config'] = config
            
            main(variant=variant)
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()