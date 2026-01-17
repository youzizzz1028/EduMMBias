#!/usr/bin/env python3
"""
API utilities for model configuration, logging, and prompt handling
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union


def load_default_config() -> Dict:
    """Load default config from configs/model_config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "default" not in config:
        raise KeyError("'default' section is missing in config file")

    return config["default"]


def load_model_config(model_key: str = None) -> Dict:
    """Load specific model configuration or all models configuration"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_key:
        if "models" not in config or model_key not in config["models"]:
            raise KeyError(f"Model '{model_key}' not found in configuration")
        return config["models"][model_key]
    else:
        return config.get("models", {})


def setup_logging(log_file: str = "llm_call.log", level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_message_prompt(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> List[Dict]:
    """Create message prompt with system and user roles"""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def create_conversation_prompt(messages: List[Dict]) -> List[Dict]:
    """Create conversation prompt from a list of message dictionaries"""
    formatted_messages = []
    
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        else:
            raise ValueError("Invalid message format. Expected dict with 'role' and 'content' keys")
    
    return formatted_messages


def format_system_prompt(system_template: str, **kwargs) -> str:
    """Format system prompt with template variables"""
    try:
        return system_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")


def format_user_prompt(user_template: str, **kwargs) -> str:
    """Format user prompt with template variables"""
    try:
        return user_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")


def get_default_config_value(key: str, default=None):
    """Get a specific value from default configuration"""
    config = load_default_config()
    return config.get(key, default)


def validate_model_config(model_config: Dict) -> bool:
    """Validate model configuration"""
    required_keys = ["model_name"]
    
    for key in required_keys:
        if key not in model_config:
            raise ValueError(f"Missing required key in model config: {key}")
    
    return True


class PromptManager:
    """Manager for handling system and user prompts"""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt"""
        self.system_prompt = system_prompt
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history"""
        self.conversation_history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to conversation history"""
        self.conversation_history.append({"role": "assistant", "content": content})
    
    def get_messages(self, user_prompt: str = None) -> List[Dict]:
        """Get formatted messages for API call"""
        messages = []
        
        # Add system prompt if set
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user prompt if provided
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()


if __name__ == "__main__":
    # Test the utilities
    try:
        # Test config loading
        default_config = load_default_config()
        print("Default config loaded successfully")
        print(f"Base URL: {default_config.get('base_url')}")
        
        # Test model config loading
        models_config = load_model_config()
        print(f"Loaded {len(models_config)} models")
        
        # Test prompt creation
        messages = create_message_prompt("Hello!", "You are a helpful AI assistant.")
        print("Message prompt created:", messages)
        
        # Test PromptManager
        pm = PromptManager("You are a helpful assistant.")
        pm.add_user_message("What is AI?")
        pm.add_assistant_message("AI is artificial intelligence.")
        messages = pm.get_messages("Tell me more about AI.")
        print("PromptManager messages:", messages)
        
    except Exception as e:
        print(f"Error testing utilities: {e}")
