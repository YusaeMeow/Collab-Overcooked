# LLM Configuration Guide

This guide explains how to configure different types of LLMs (Large Language Models) in Collab-Overcooked, including OpenAI models, local vLLM deployment, and custom API services.

## Overview

Collab-Overcooked supports three types of LLM configurations:

1. **OpenAI Models** (GPT-3.5, GPT-4, etc.)
2. **Local vLLM Deployment** (Llama, CodeLlama, etc.)
3. **Custom API Services** (DeepSeek, Claude, etc.)

## Quick Start

### 1. OpenAI Models (Recommended for beginners)

```yaml
agents:
  agent_0:
    type: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

**Setup:**
1. Get an OpenAI API key from https://platform.openai.com
2. Save it to `configs/openai_key.txt`
3. Run: `collab-overcooked --config configs/default.yaml`

### 2. Local vLLM Deployment (Cost-effective for heavy usage)

```yaml
agents:
  agent_0:
    type: "vllm"
    model: "meta-llama/Llama-2-7b-chat-hf"
    base_url: "http://localhost:8000/v1"
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

**Setup:**
1. Install vLLM: `pip install vllm`
2. Start server: `bash scripts/start_vllm.sh`
3. Run: `collab-overcooked --config configs/examples/vllm_config.yaml`

### 3. Custom API Services (DeepSeek, Claude, etc.)

```yaml
agents:
  agent_0:
    type: "custom_api"
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"
    api_key: "your_api_key"
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

## Detailed Configuration

### OpenAI Configuration

```yaml
agents:
  agent_0:
    type: "openai"                    # Required: Use OpenAI API
    model: "gpt-3.5-turbo"           # Model name
    temperature: 0.1                 # Randomness (0.0-2.0)
    max_tokens: 512                  # Max response length
    timeout: 30                      # Request timeout (seconds)
    max_retries: 3                   # Retry attempts
    api_key: "sk-..."               # Optional: API key (uses config file if not provided)
    role: "Chef"                     # Agent role
```

**Supported Models:**
- `gpt-3.5-turbo` (recommended for most tasks)
- `gpt-4` (better quality, slower)
- `gpt-4-turbo` (faster GPT-4)
- `gpt-4o` (latest model)

### vLLM Configuration

```yaml
agents:
  agent_0:
    type: "vllm"                                        # Required: Use vLLM
    model: "meta-llama/Llama-2-7b-chat-hf"            # HuggingFace model name
    base_url: "http://localhost:8000/v1"               # vLLM server URL
    model_path: "/path/to/local/model"                 # Optional: local model path for tokenizer
    temperature: 0.1                                   # Randomness
    max_tokens: 512                                    # Max response length
    timeout: 60                                        # Longer timeout for local models
    max_retries: 3                                     # Retry attempts
    role: "Chef"                                       # Agent role
```

**Popular Models:**
- `meta-llama/Llama-2-7b-chat-hf` (good balance)
- `meta-llama/Llama-2-13b-chat-hf` (better quality)
- `meta-llama/CodeLlama-7b-Instruct-hf` (code-focused)
- `mistralai/Mistral-7B-Instruct-v0.1` (fast and capable)

### Custom API Configuration

```yaml
agents:
  agent_0:
    type: "custom_api"                           # Required: Use custom API
    model: "deepseek-chat"                       # Model name
    base_url: "https://api.deepseek.com/v1"     # API endpoint
    api_key: "your_api_key"                     # API key
    temperature: 0.1                            # Randomness
    max_tokens: 512                             # Max response length
    timeout: 30                                 # Request timeout
    max_retries: 3                              # Retry attempts
    role: "Chef"                                # Agent role
```

**Example Services:**
- **DeepSeek**: `https://api.deepseek.com/v1`
- **Anthropic Claude**: `https://api.anthropic.com/v1`
- **Together AI**: `https://api.together.xyz/v1`

## Advanced Configuration

### Mixed Agent Setup

Use different models for different agents:

```yaml
agents:
  agent_0:
    type: "openai"
    model: "gpt-4"                    # Smart planner
    temperature: 0.0                  # Deterministic
    role: "Chef"
  
  agent_1:
    type: "vllm"
    model: "meta-llama/Llama-2-13b-chat-hf"  # Cost-effective executor
    base_url: "http://localhost:8000/v1"
    temperature: 0.2                  # More creative
    role: "Assistant"
```

### Multiple vLLM Servers

Run different models on different ports:

```bash
# Terminal 1: 7B model for assistant
python -m vllm.entrypoints.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000

# Terminal 2: 13B model for chef  
python -m vllm.entrypoints.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --port 8001
```

```yaml
agents:
  agent_0:
    type: "vllm"
    model: "meta-llama/Llama-2-13b-chat-hf"
    base_url: "http://localhost:8001/v1"    # Larger model for chef
    role: "Chef"
  
  agent_1:
    type: "vllm"
    model: "meta-llama/Llama-2-7b-chat-hf"
    base_url: "http://localhost:8000/v1"    # Smaller model for assistant
    role: "Assistant"
```

## Setup Instructions

### vLLM Setup

1. **Install vLLM:**
   ```bash
   pip install vllm
   ```

2. **Start vLLM server:**
   ```bash
   # Using provided script
   bash scripts/start_vllm.sh meta-llama/Llama-2-7b-chat-hf 8000
   
   # Or manually
   python -m vllm.entrypoints.api_server \
     --model meta-llama/Llama-2-7b-chat-hf \
     --port 8000 \
     --gpu-memory-utilization 0.8
   ```

3. **Test connection:**
   ```bash
   python scripts/test_llm_setup.py
   ```

### Custom API Setup

1. **Get API credentials** from your provider
2. **Set environment variables** (optional):
   ```bash
   export DEEPSEEK_API_KEY="your_key"
   export ANTHROPIC_API_KEY="your_key"
   ```

3. **Configure in YAML:**
   ```yaml
   agents:
     agent_0:
       type: "custom_api"
       model: "deepseek-chat"
       base_url: "https://api.deepseek.com/v1"
       api_key: "${DEEPSEEK_API_KEY}"  # Use environment variable
   ```

## Configuration Examples

Complete example configurations are available in:

- `configs/examples/vllm_config.yaml` - Local vLLM setup
- `configs/examples/mixed_agents.yaml` - Mixed model types
- `configs/examples/custom_api.yaml` - Custom API services

## Testing Your Setup

Use the provided test script to verify your configuration:

```bash
python scripts/test_llm_setup.py
```

This will test:
- Configuration file loading
- Token counting
- API connections (if credentials provided)

## Performance Tips

### Cost Optimization

1. **Use local models for development:**
   ```yaml
   # Development: Fast local model
   type: "vllm"
   model: "meta-llama/Llama-2-7b-chat-hf"
   ```

2. **Use API models for production:**
   ```yaml
   # Production: High-quality API model
   type: "openai"
   model: "gpt-4"
   ```

### Speed Optimization

1. **Lower temperature for deterministic responses:**
   ```yaml
   temperature: 0.0  # Fastest, most deterministic
   ```

2. **Reduce max_tokens for shorter responses:**
   ```yaml
   max_tokens: 256  # Faster generation
   ```

3. **Use smaller models for simple tasks:**
   ```yaml
   model: "gpt-3.5-turbo"  # vs gpt-4
   model: "meta-llama/Llama-2-7b-chat-hf"  # vs 13b
   ```

### Memory Optimization (vLLM)

1. **Adjust GPU memory utilization:**
   ```bash
   python -m vllm.entrypoints.api_server \
     --model meta-llama/Llama-2-7b-chat-hf \
     --gpu-memory-utilization 0.6  # Use 60% of GPU memory
   ```

2. **Use quantized models:**
   ```bash
   python -m vllm.entrypoints.api_server \
     --model TheBloke/Llama-2-7B-Chat-GPTQ \
     --quantization gptq
   ```

## Troubleshooting

### Common Issues

1. **OpenAI API key not found:**
   - Solution: Add key to `configs/openai_key.txt`

2. **vLLM server connection failed:**
   - Check if server is running: `curl http://localhost:8000/v1/models`
   - Check firewall/port settings

3. **Out of memory errors (vLLM):**
   - Use smaller model: `Llama-2-7b` instead of `Llama-2-13b`
   - Reduce GPU memory utilization: `--gpu-memory-utilization 0.5`

4. **Custom API authentication failed:**
   - Verify API key and base URL
   - Check API documentation for correct endpoint format

### Debug Mode

Enable debug output to see detailed API calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set debug=True in agent creation:

```python
response = agent.query("test input", debug=True)
```

## Migration from Old System

If you're upgrading from the old modules.py system:

1. **Old configuration:**
   ```python
   Module(
       role_messages,
       model="gpt-3.5-turbo",
       local_server_api="http://localhost:8000/v1"
   )
   ```

2. **New configuration:**
   ```yaml
   agents:
     agent_0:
       type: "openai"  # or "vllm"
       model: "gpt-3.5-turbo"
       base_url: "http://localhost:8000/v1"  # only for vLLM/custom
   ```

The new system provides:
- ✅ Cleaner configuration
- ✅ Better error handling
- ✅ Automatic token counting
- ✅ Type safety
- ✅ Easy switching between model types