"""
Refactored modules.py - Clean LLM interface without key file dependencies
"""

import time
import pandas as pd
import numpy as np
from scipy import spatial
from typing import Union, Dict, List, Optional
from rich import print as rprint
import tiktoken
from openai import OpenAI
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
from .web_util import output_to_port

# Global statistics
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
}

# Turn statistics
turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        "original_log": "",
    },
}

# Token limits for different models
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4o": 8192,
    "llama3:70b-instruct-fp16": 4096,
}

EMBEDDING_MODEL = "text-embedding-3-small"


class Module:
    """
    Simplified LLM communication module that gets API keys from agent configuration
    """

    def __init__(
        self,
        role_messages,
        model="gpt-3.5-turbo",
        api_key=None,
        base_url=None,
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
        retrival_method="recent_k",
        K=3,
    ):
        self.model = model
        self.api_key = api_key  # Now provided by agent config
        self.base_url = base_url or self._get_default_base_url()
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.retrival_method = retrival_method
        self.K = K

        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.dialog_history_list_storage = []
        self.current_user_message = None
        self.cache_list = None
        self.experience = []
        self.embedding = None
        self.current_timestep = None

    def _get_default_base_url(self):
        """Get default base URL based on model type"""
        if "gpt" in self.model or "deepseek" in self.model.lower():
            return "https://api.openai.com/v1"
        else:
            return self.local_server_api

    def _get_client(self):
        """Create OpenAI client with proper configuration"""
        if not self.api_key:
            raise ValueError("API key is required but not provided")
        
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def load_embedding(self):
        """Load embeddings for similarity matching"""
        try:
            df = pd.read_csv(f"data/embedding_{self.name.lower()}.csv")
            df["embedding"] = df.embedding.apply(eval).apply(np.array)
            self.embedding = df
        except FileNotFoundError:
            rprint("[yellow]Warning: Embedding file not found[/yellow]")
            self.embedding = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        """Add messages to instruction head"""
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        """Add message to dialog history"""
        self.dialog_history_list.append(message)

    def get_cache(self) -> list:
        """Get cached messages based on retrieval method"""
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K :]
            else:
                return []
        return []

    def query_messages(self, rethink=False) -> list:
        """Build query messages"""
        system_message = {
            "role": "system",
            "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
        }
        
        user_content = (
            self.instruction_head_list[0]["content"] + 
            "<input>\n" + 
            self.current_user_message["content"]
        )
        
        return [
            system_message,
            {"role": "user", "content": user_content}
        ]

    @retry_with_exponential_backoff
    def query(
        self,
        key=None,  # Deprecated, kept for compatibility
        proxy=None,  # Deprecated, kept for compatibility
        stop=None,
        temperature=0.7,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        """
        Query LLM with simplified model handling
        """
        messages = self.query_messages(rethink)
        self.cache_list = self.get_cache()

        if not trace and not rethink:
            messages[-1]["content"] += " Based on the failure explanation and scene description, analyze and plan again."

        response = None
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                if "human" in self.model:
                    # Human interface (web-based)
                    response = self._handle_human_interface(messages, map)
                    encoder_name = "gpt-3.5-turbo"
                    
                elif self.model in ["text-davinci-003"]:
                    # Legacy completion model
                    response = self._handle_completion_model(messages, temperature, stop)
                    encoder_name = "p50k_base"
                    
                else:
                    # All chat models (GPT, DeepSeek, vLLM, etc.)
                    response = self._handle_chat_model(messages, temperature)
                    encoder_name = self._get_encoder_name()

                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                rprint(f"[red][LLM ERROR][/red]: {e}")
                if retry_count >= max_retries:
                    rprint("[red][ERROR][/red]: Query failed after maximum retries!")
                    return "", 0
                time.sleep(1)

        response_text = self.parse_response(response)
        token_count = self._count_tokens(response_text, encoder_name)
        
        return response_text, token_count

    def _handle_human_interface(self, messages, map):
        """Handle human interface through web"""
        content = messages[-1]["content"]
        
        # Determine receiver
        if "Suppose you are a Chef" in content:
            receiver = "agent0"
        elif "Suppose you are a Assistant" in content:
            receiver = "agent1"
        else:
            raise ValueError("Invalid role in human interface")
        
        # Extract relevant parts
        input_start = content.find("<input>\n") + len("<input>\n")
        human_message = content[input_start:]
        
        # Extract recipe for chef
        recipe = None
        if receiver == "agent0":
            recipe_start = content.find("<Recipe need to know>:\n")
            if recipe_start != -1:
                recipe_start += len("<Recipe need to know>:\n")
                recipe_end = content.find("**Skill**")
                recipe = content[recipe_start:recipe_end]
        
        # Extract error information
        error = None
        error_start = human_message.find("DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n")
        if error_start != -1:
            error = human_message[error_start + len("DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n"):]
            human_message = human_message[:human_message.find("Below are the failed and analysis history")]
        
        return output_to_port(receiver, human_message, map=map, recipe=recipe, error=error)

    def _handle_completion_model(self, messages, temperature, stop):
        """Handle completion models like text-davinci-003"""
        import openai  # Legacy openai for completion
        
        prompt = convert_messages_to_prompt(messages)
        return openai.Completion.create(
            model=self.model,
            prompt=prompt,
            stop=stop,
            temperature=temperature,
            max_tokens=256,
        )

    def _handle_chat_model(self, messages, temperature):
        """Handle all chat models (GPT, DeepSeek, vLLM, etc.)"""
        client = self._get_client()
        
        # For vLLM and local models, use full model path
        model_name = self.model
        if not ("gpt" in self.model or "deepseek" in self.model.lower()):
            model_name = self.model_dirname + self.model
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        
        time.sleep(0.5)  # Rate limiting
        return response

    def _get_encoder_name(self):
        """Get encoder name for token counting"""
        if "gpt-4" in self.model:
            return "gpt-4"
        elif "gpt-3.5" in self.model:
            return "gpt-3.5-turbo"
        elif "deepseek" in self.model.lower():
            return "gpt-4"  # Use GPT-4 tokenizer for DeepSeek
        else:
            return "llama3"  # For other models

    def _count_tokens(self, text, encoder_name):
        """Count tokens in response"""
        try:
            if "gpt" in encoder_name:
                encoding = tiktoken.encoding_for_model(encoder_name)
                return len(encoding.encode(text))
            elif "llama3" in encoder_name:
                # Use local tokenizer if available
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "../lib/llama_tokenizer", local_files_only=True
                    )
                    return len(tokenizer.encode(text))
                except:
                    # Fallback: estimate tokens
                    return len(text.split()) * 1.3  # Rough estimation
            else:
                return len(text.split()) * 1.2  # Rough estimation
        except Exception:
            return len(text.split())  # Basic fallback

    def parse_response(self, response):
        """Parse response from different model types"""
        if "human" in self.model:
            return self._parse_human_response(response)
        elif self.model in ["text-davinci-003"]:
            return response["choices"][0]["text"]
        elif hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
            return response.choices[0].message.content
        elif isinstance(response, dict):
            if "choices" in response:
                if "message" in response["choices"][0]:
                    return response["choices"][0]["message"]["content"]
                elif "content" in response["choices"][0]:
                    return response["choices"][0]["content"]
        
        # Fallback
        return str(response)

    def _parse_human_response(self, response):
        """Parse human interface response"""
        role = "Assistant" if response.get("agent") == "agent1" else "Chef"
        plan = response.get("plan", "")
        say = response.get("say", "")
        
        return f"{role} analysis: [NOTHING]\n{role} plan: {plan}\n{role} say: {say if say else '[NOTHING]'}"

    def reset(self):
        """Reset dialog history"""
        self.dialog_history_list = []

    def get_top_k_similar_example(self, key=None, k=4):
        """Get similar examples using embeddings (deprecated key parameter)"""
        if k == 0 or not self.api_key:
            return ""
        
        try:
            client = self._get_client()
            input_text = self.current_user_message["content"]
            
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, 
                input=[input_text]
            )
            
            input_embedding = response.data[0].embedding
            
            if self.embedding is None:
                self.load_embedding()
                if self.embedding is None:
                    return ""
            
            self.embedding["similarities"] = self.embedding.embedding.apply(
                lambda x: 1 - spatial.distance.cosine(x, input_embedding)
            )
            
            top_k_strings = self.embedding.sort_values(
                "similarities", ascending=False
            ).head(k)["text"]
            
            result = ""
            for t in top_k_strings:
                if t.startswith("\n"):
                    t = t[1:]
                result += f"<example>\n{t}\n</example>\n\n"
            
            prompt_begin = {
                "Chef": "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment. You should make plan for yourself in 'Chef plan', and make plan for assistant by saying to him.\n",
                "Assistant": "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment. If you do not know what to do, just ask chef to make a plan for you.\n"
            }
            
            return prompt_begin.get(getattr(self, 'name', 'Chef'), "") + result
            
        except Exception as e:
            rprint(f"[yellow]Warning: Failed to get similar examples: {e}[/yellow]")
            return ""


def if_two_sentence_similar_meaning(key=None, proxy=None, sentence1="", sentence2=""):
    """
    Check if two sentences have similar meaning using embeddings
    Note: This function is deprecated and should use agent's API key
    """
    if not sentence1 or not sentence2:
        return False
    
    # This function needs to be called with an API key from the agent
    rprint("[yellow]Warning: if_two_sentence_similar_meaning needs API key from agent[/yellow]")
    return False