#!/usr/bin/env python
"""
Examples showing how to configure different LLM types in Collab-Overcooked
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collab_overcooked.agents import (
    LLMConfig, 
    LLMClientFactory, 
    create_llm_config_from_yaml,
    PRESET_CONFIGS,
    CollaborativeAgent
)


def example_openai_config():
    """Example: Configure OpenAI GPT models"""
    
    print("=== OpenAI Configuration Examples ===")
    
    # Method 1: Using LLMConfig directly
    config1 = LLMConfig(
        model_type="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512,
        api_key="your_api_key_here"  # Will use config file if not provided
    )
    
    # Method 2: Using preset configurations
    config2 = LLMConfig(**PRESET_CONFIGS["gpt-4"])
    
    # Method 3: From YAML-style dict (common in config files)
    yaml_config = {
        "type": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 512
    }
    config3 = create_llm_config_from_yaml(yaml_config)
    
    print(f"Config 1 - Model: {config1.model_name}, Type: {config1.model_type}")
    print(f"Config 2 - Model: {config2.model_name}, Type: {config2.model_type}")
    print(f"Config 3 - Model: {config3.model_name}, Type: {config3.model_type}")


def example_vllm_config():
    """Example: Configure local vLLM deployment"""
    
    print("\n=== vLLM Configuration Examples ===")
    
    # Method 1: Basic vLLM configuration
    config1 = LLMConfig(
        model_type="vllm",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        base_url="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=512
    )
    
    # Method 2: With local model path for tokenizer
    config2 = LLMConfig(
        model_type="vllm",
        model_name="meta-llama/Llama-2-13b-chat-hf",
        base_url="http://localhost:8001/v1",
        model_path="/path/to/llama2-13b-chat",  # For proper token counting
        temperature=0.2,
        max_tokens=1024,
        timeout=60  # Longer timeout for larger models
    )
    
    # Method 3: From YAML-style dict
    yaml_config = {
        "type": "vllm",
        "model": "meta-llama/CodeLlama-7b-Instruct-hf",
        "base_url": "http://192.168.1.100:8000/v1",  # Remote vLLM server
        "temperature": 0.0,
        "max_tokens": 512
    }
    config3 = create_llm_config_from_yaml(yaml_config)
    
    print(f"Config 1 - Model: {config1.model_name}, URL: {config1.base_url}")
    print(f"Config 2 - Model: {config2.model_name}, Timeout: {config2.timeout}s")
    print(f"Config 3 - Model: {config3.model_name}, URL: {config3.base_url}")


def example_custom_api_config():
    """Example: Configure custom API services (DeepSeek, Claude, etc.)"""
    
    print("\n=== Custom API Configuration Examples ===")
    
    # DeepSeek API
    deepseek_config = LLMConfig(
        model_type="custom_api",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="your_deepseek_key",
        temperature=0.1,
        max_tokens=512
    )
    
    # Claude API (Anthropic)
    claude_config = LLMConfig(
        model_type="custom_api",
        model_name="claude-3-sonnet-20240229",
        base_url="https://api.anthropic.com/v1",
        api_key="your_anthropic_key",
        temperature=0.1,
        max_tokens=512
    )
    
    # From YAML with automatic DeepSeek URL
    yaml_config = {
        "type": "deepseek",  # Special shortcut for DeepSeek
        "model": "deepseek-reasoner",
        "api_key": "your_deepseek_key",
        "temperature": 0.2,
        "max_tokens": 1024
    }
    auto_config = create_llm_config_from_yaml(yaml_config)
    
    print(f"DeepSeek - Model: {deepseek_config.model_name}")
    print(f"Claude - Model: {claude_config.model_name}")
    print(f"Auto DeepSeek - URL: {auto_config.base_url}")


def example_agent_creation():
    """Example: Create agents with different LLM configurations"""
    
    print("\n=== Agent Creation Examples ===")
    
    # Create different agent configurations
    configs = {
        "chef_gpt4": {
            "type": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 512,
            "role": "Chef"
        },
        "assistant_llama": {
            "type": "vllm",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "base_url": "http://localhost:8000/v1",
            "temperature": 0.2,
            "max_tokens": 512,
            "role": "Assistant"
        },
        "chef_deepseek": {
            "type": "custom_api",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "your_key",
            "temperature": 0.1,
            "max_tokens": 512,
            "role": "Chef"
        }
    }
    
    # Create agents (would work with proper API keys/servers)
    for name, config in configs.items():
        try:
            # Note: This would fail without proper API keys/servers
            # agent = CollaborativeAgent(config)
            # print(f"Created {name}: {agent.agent_module.llm_config.model_name}")
            print(f"Config for {name}: {config['model']} ({config['type']})")
        except Exception as e:
            print(f"Config for {name}: {config['model']} ({config['type']}) - {e}")


def example_mixed_setup():
    """Example: Mixed setup with different model types"""
    
    print("\n=== Mixed Setup Example ===")
    
    # Configuration for a mixed agent setup
    mixed_config = {
        "environment": {
            "horizon": 15,
            "order": "soup",
            "layout": "coordination_ring"
        },
        "agents": {
            "agent_0": {
                "type": "openai",
                "model": "gpt-4",
                "temperature": 0.0,
                "role": "Chef"
            },
            "agent_1": {
                "type": "vllm", 
                "model": "meta-llama/Llama-2-13b-chat-hf",
                "base_url": "http://localhost:8000/v1",
                "temperature": 0.2,
                "role": "Assistant"
            }
        }
    }
    
    print("Mixed setup configuration:")
    for agent_id, agent_config in mixed_config["agents"].items():
        llm_config = create_llm_config_from_yaml(agent_config)
        print(f"  {agent_id}: {llm_config.model_name} ({llm_config.model_type})")
    
    print("\nThis allows:")
    print("- Chef using GPT-4 for strategic planning")
    print("- Assistant using local Llama for cost-effective execution")
    print("- Easy switching between different model combinations")


def show_usage_summary():
    """Show usage summary"""
    
    print("\n" + "="*50)
    print("USAGE SUMMARY")
    print("="*50)
    
    print("\n1. OpenAI Models (GPT-3.5, GPT-4):")
    print("   - Set API key in configs/openai_key.txt")
    print("   - type: 'openai'")
    print("   - Automatic token counting and rate limiting")
    
    print("\n2. Local vLLM Deployment:")
    print("   - Start vLLM server: python -m vllm.entrypoints.api_server \\")
    print("     --model meta-llama/Llama-2-7b-chat-hf --port 8000")
    print("   - type: 'vllm'")
    print("   - base_url: 'http://localhost:8000/v1'")
    
    print("\n3. Custom API Services:")
    print("   - type: 'custom_api'")
    print("   - Provide both api_key and base_url")
    print("   - Works with DeepSeek, Claude, etc.")
    
    print("\n4. Configuration Methods:")
    print("   - YAML config files (recommended)")
    print("   - Python dict -> create_llm_config_from_yaml()")
    print("   - Direct LLMConfig() instantiation")
    
    print("\n5. Quick Start:")
    print("   - Copy configs/examples/vllm_config.yaml")
    print("   - Modify for your setup")
    print("   - Run: collab-overcooked --config your_config.yaml")


if __name__ == "__main__":
    print("Collab-Overcooked LLM Configuration Examples")
    print("="*50)
    
    example_openai_config()
    example_vllm_config()
    example_custom_api_config()
    example_agent_creation()
    example_mixed_setup()
    show_usage_summary()