# Configuration Guide

## YAML Configuration (Recommended)

All configuration should now be done through YAML files. API keys should be included directly in the YAML configuration.

### Quick Start

1. **Use default config**: `configs/default.yaml` contains a template configuration
2. **Personal config**: Copy `default.yaml` to `test_personal.yaml` and add your API key
3. **Run tests**: Use `scripts/quick_test.sh` which automatically detects available configs

### Running the System

You have multiple ways to run the system:

1. **Quick test script** (recommended):
   ```bash
   bash scripts/quick_test.sh
   ```

2. **Direct Python call**:
   ```bash
   python -c "from collab_overcooked.main import main; main(config_path='configs/test_personal.yaml')"
   ```

3. **CLI module** (if installed):
   ```bash
   python -m collab_overcooked.cli --config configs/test_personal.yaml
   ```

4. **Installed command** (after pip install):
   ```bash
   collab-overcooked --config configs/test_personal.yaml
   ```

### Configuration Priority

1. `configs/test_personal.yaml` (highest priority)
2. `configs/default.yaml` (default fallback)
3. Other `*.yaml` files in configs/

### Configuration Types

#### OpenAI API
```yaml
agents:
  agent_0:
    type: "openai"
    model: "gpt-3.5-turbo"
    api_key: "your_openai_api_key_here"
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

#### Custom API (like your current setup)
```yaml
agents:
  agent_0:
    type: "custom_api"
    model: "gpt-3.5-turbo"
    base_url: "https://api2.aigcbest.top/v1"
    api_key: "your_custom_api_key_here"
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

#### Local vLLM
```yaml
agents:
  agent_0:
    type: "vllm"
    model: "llama-2-7b-chat"
    base_url: "http://localhost:8000/v1"  # Optional, defaults to this
    temperature: 0.1
    max_tokens: 512
    role: "Chef"
```

### Environment Variables (Alternative)

You can also set API keys via environment variables:
- `OPENAI_API_KEY`: For OpenAI or custom API endpoints
- `API_KEY`: Generic fallback

### Legacy Support

The system still supports legacy key files as fallback:
- `configs/personal_api_key.txt`
- `configs/openai_key.txt`

However, YAML configuration is recommended for better organization and security.

### Security Notes

- Add `*.yaml` files with real API keys to `.gitignore`
- Use `configs/default.yaml` as a template with placeholder keys
- Consider using environment variables for production deployments