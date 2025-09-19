#!/usr/bin/env python
"""
Test script to verify LLM configurations work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install PyYAML")
    sys.exit(1)

from collab_overcooked.agents import LLMClientFactory, create_llm_config_from_yaml


def test_openai_config():
    """Test OpenAI configuration"""
    print("Testing OpenAI configuration...")
    
    try:
        config = {
            "type": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 50  # Small for testing
        }
        
        llm_config = create_llm_config_from_yaml(config)
        client = LLMClientFactory.create_client(llm_config)
        
        messages = [{"role": "user", "content": "Say 'Hello from OpenAI!'"}]
        response, tokens = client.generate(messages)
        
        print(f"✅ OpenAI test successful!")
        print(f"   Response: {response[:50]}...")
        print(f"   Tokens: {tokens}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False


def test_vllm_config():
    """Test vLLM configuration"""
    print("\nTesting vLLM configuration...")
    
    try:
        config = {
            "type": "vllm",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "base_url": "http://localhost:8000/v1",
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        llm_config = create_llm_config_from_yaml(config)
        client = LLMClientFactory.create_client(llm_config)
        
        messages = [{"role": "user", "content": "Say 'Hello from vLLM!'"}]
        response, tokens = client.generate(messages)
        
        print(f"✅ vLLM test successful!")
        print(f"   Response: {response[:50]}...")
        print(f"   Tokens: {tokens}")
        return True
        
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        print(f"   Make sure vLLM server is running on localhost:8000")
        return False


def test_custom_api_config():
    """Test custom API configuration"""
    print("\nTesting custom API configuration...")
    
    try:
        config = {
            "type": "custom_api",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "dummy_key",  # This will fail but test config loading
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        llm_config = create_llm_config_from_yaml(config)
        # Note: We only test config creation, not actual API call
        print(f"✅ Custom API config loading successful!")
        print(f"   Model: {llm_config.model_name}")
        print(f"   URL: {llm_config.base_url}")
        print(f"   (API call not tested - requires valid key)")
        return True
        
    except Exception as e:
        print(f"❌ Custom API test failed: {e}")
        return False


def test_config_file_loading():
    """Test loading configuration from YAML files"""
    print("\nTesting config file loading...")
    
    try:
        # Test loading default config
        config_path = "configs/default.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for agent_id, agent_config in config['agents'].items():
                if agent_id.startswith('agent_'):
                    llm_config = create_llm_config_from_yaml(agent_config)
                    print(f"✅ Loaded config for {agent_id}: {llm_config.model_name}")
            
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Config file loading failed: {e}")
        return False


def test_token_counting():
    """Test token counting functionality"""
    print("\nTesting token counting...")
    
    try:
        # Test OpenAI tokenizer
        config = create_llm_config_from_yaml({
            "type": "openai",
            "model": "gpt-3.5-turbo"
        })
        client = LLMClientFactory.create_client(config)
        
        test_text = "This is a test message for token counting."
        tokens = client.count_tokens(test_text)
        
        print(f"✅ Token counting works!")
        print(f"   Text: '{test_text}'")
        print(f"   Tokens: {tokens}")
        return True
        
    except Exception as e:
        print(f"❌ Token counting failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Collab-Overcooked LLM Setup Test")
    print("=" * 40)
    
    results = []
    
    # Basic configuration tests
    results.append(test_config_file_loading())
    results.append(test_token_counting())
    results.append(test_custom_api_config())
    
    # API tests (these may fail without proper setup)
    print("\n" + "=" * 40)
    print("API Connection Tests (may fail without proper setup)")
    print("=" * 40)
    
    results.append(test_openai_config())
    results.append(test_vllm_config())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Your LLM setup is ready.")
    else:
        print("⚠️  Some tests failed. Check the configuration and API keys.")
        print("\nTroubleshooting:")
        print("- For OpenAI: Set API key in configs/openai_key.txt")
        print("- For vLLM: Start server with scripts/start_vllm.sh")
        print("- For custom APIs: Set proper API keys and URLs")


if __name__ == "__main__":
    main()