"""
LLM Client module for supporting both local vLLM and API-based LLMs
"""

import os
import time
import tiktoken
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from openai import OpenAI
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
import logging

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration class for LLM settings"""
    
    def __init__(
        self,
        model_type: str = "openai",  # "openai", "vllm", "custom_api"
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_path: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.model_path = model_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_params = kwargs
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        # For vLLM, set default base URL if not provided
        if self.model_type == "vllm" and not self.base_url:
            self.base_url = "http://localhost:8000/v1"
        
        # For custom API, base URL is required
        if self.model_type == "custom_api" and not self.base_url:
            raise ValueError("Base URL is required for custom API models")
        
        # API key validation for all non-vLLM models
        if self.model_type in ["openai", "custom_api"] and not self.api_key:
            # Try to load from environment variable first
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if env_key:
                self.api_key = env_key
                return
            
            # Try to load from legacy config files as fallback
            possible_key_files = [
                "configs/personal_api_key.txt",
                "configs/openai_key.txt",
                "configs/custom_api_key.txt"
            ]
            
            for key_file in possible_key_files:
                config_path = os.path.join(os.getcwd(), key_file)
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        self.api_key = f.read().strip().split("\n")[0]
                    logger.warning(f"Using legacy API key file: {key_file}. Consider migrating to YAML config.")
                    break
            
            if not self.api_key:
                raise ValueError(
                    f"API key is required for {self.model_type} models. "
                    "Please provide api_key in YAML config or set OPENAI_API_KEY environment variable."
                )


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self):
        """Setup the LLM client"""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Generate response from messages
        
        Returns:
            tuple: (response_text, token_count)
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(text.split())  # Default simple implementation


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
    }
    
    def _setup_client(self):
        """Setup OpenAI client"""
        self.client = OpenAI(api_key=self.config.api_key)
        
        # Setup tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            # Fallback to a default encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Generate response using OpenAI API"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    **self.config.extra_params
                )
                
                content = response.choices[0].message.content
                token_count = self.count_tokens(content)
                
                # Rate limiting
                time.sleep(0.5)
                
                return content, token_count
                
            except Exception as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))


class VLLMClient(BaseLLMClient):
    """vLLM local server client"""
    
    def _setup_client(self):
        """Setup vLLM client"""
        self.client = OpenAI(
            api_key="token-abc123",  # vLLM doesn't need real API key
            base_url=self.config.base_url
        )
        
        # Setup tokenizer for token counting if model_path is provided
        if self.config.model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {self.config.model_path}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Generate response using vLLM server"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    **self.config.extra_params
                )
                
                content = response.choices[0].message.content
                token_count = self.count_tokens(content)
                
                return content, token_count
                
            except Exception as e:
                logger.error(f"vLLM API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using model tokenizer"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())  # Fallback to word count


class CustomAPIClient(BaseLLMClient):
    """Custom API client for other LLM services (DeepSeek, etc.)"""
    
    def _setup_client(self):
        """Setup custom API client"""
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Generate response using custom API"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    **self.config.extra_params
                )
                
                content = response.choices[0].message.content
                token_count = self.count_tokens(content)
                
                return content, token_count
                
            except Exception as e:
                logger.error(f"Custom API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1)


class LLMClientFactory:
    """Factory class for creating LLM clients"""
    
    @staticmethod
    def create_client(config: Union[Dict, LLMConfig]) -> BaseLLMClient:
        """Create LLM client based on configuration"""
        if isinstance(config, dict):
            config = LLMConfig(**config)
        
        if config.model_type == "openai":
            return OpenAIClient(config)
        elif config.model_type == "vllm":
            return VLLMClient(config)
        elif config.model_type == "custom_api":
            return CustomAPIClient(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")


def create_llm_config_from_yaml(agent_config: Dict[str, Any]) -> LLMConfig:
    """Create LLM config from YAML agent configuration"""
    
    # Map old format to new format
    model_type_mapping = {
        "gpt": "openai",
        "openai": "openai",
        "vllm": "vllm",
        "local_llm": "vllm",
        "custom_api": "custom_api",
        "deepseek": "custom_api",
    }
    
    agent_type = agent_config.get("type", "gpt")
    model_type = model_type_mapping.get(agent_type, "openai")
    
    config_params = {
        "model_type": model_type,
        "model_name": agent_config.get("model", "gpt-3.5-turbo"),
        "temperature": agent_config.get("temperature", 0.1),
        "max_tokens": agent_config.get("max_tokens", 512),
        "api_key": agent_config.get("api_key"),
        "base_url": agent_config.get("base_url"),
        "model_path": agent_config.get("model_path"),
        "timeout": agent_config.get("timeout", 30),
        "max_retries": agent_config.get("max_retries", 3),
    }
    
    # Handle specific model types
    if model_type == "custom_api" and agent_type == "deepseek":
        if not config_params["base_url"]:
            config_params["base_url"] = "https://api.deepseek.com/v1"
    
    # Remove None values
    config_params = {k: v for k, v in config_params.items() if v is not None}
    
    return LLMConfig(**config_params)


# Example usage and configuration presets
PRESET_CONFIGS = {
    "gpt-3.5-turbo": {
        "model_type": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 512,
    },
    "gpt-4": {
        "model_type": "openai", 
        "model_name": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 512,
    },
    "local-llama": {
        "model_type": "vllm",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.1,
        "max_tokens": 512,
    },
    "deepseek": {
        "model_type": "custom_api",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "temperature": 0.1,
        "max_tokens": 512,
    }
}