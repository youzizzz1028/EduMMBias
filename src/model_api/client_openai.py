#src/model_api/client_openai.py
from openai import OpenAI
import logging
import time
from . import api_utils


def create_openai_client(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    log: str = None,
    max_retries: int = None,
    retry_delay: float = None,
    timeout: int = None,
):
    """
    Create an OpenAI client.
    `model_name` is required.
    Other arguments fall back to default config values if not provided.
    """

    if not model_name:
        raise ValueError("model_name is required")

    # Load default config from YAML using api_utils
    default_cfg = api_utils.load_default_config()

    base_url = base_url or default_cfg.get("base_url")
    api_key = api_key or default_cfg.get("api_key")
    log_file = log or default_cfg.get("log", "llm_call.log")
    max_retries = max_retries or default_cfg.get("max_retries", 3)
    retry_delay = retry_delay or default_cfg.get("retry_delay", 1)
    timeout = timeout or default_cfg.get("timeout", 60)

    # Setup logging using api_utils
    api_utils.setup_logging(log_file=log_file)

    # Create OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )

    def call(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Call the model with retry logic"""
        # Create messages using api_utils
        messages = api_utils.create_message_prompt(prompt, system_prompt)
        
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                logging.info(f"Model call succeeded: model={model_name}")
                msg = response.choices[0].message
                return getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else None)

            except Exception as e:
                logging.error(f"Model call failed (attempt {attempt}): {e}")

                if attempt == max_retries:
                    raise

                time.sleep(retry_delay)

    return client, call


def create_openai_client_with_prompt_manager(
    model_name: str,
    prompt_manager: api_utils.PromptManager = None,
    base_url: str = None,
    api_key: str = None,
    log: str = None,
    max_retries: int = None,
    retry_delay: float = None,
    timeout: int = None,
):
    """
    Create an OpenAI client with PromptManager support for conversation history.
    """
    if not model_name:
        raise ValueError("model_name is required")

    # Load default config from YAML using api_utils
    default_cfg = api_utils.load_default_config()

    base_url = base_url or default_cfg.get("base_url")
    api_key = api_key or default_cfg.get("api_key")
    log_file = log or default_cfg.get("log", "llm_call.log")
    max_retries = max_retries or default_cfg.get("max_retries", 3)
    retry_delay = retry_delay or default_cfg.get("retry_delay", 1)
    timeout = timeout or default_cfg.get("timeout", 60)

    # Setup logging using api_utils
    api_utils.setup_logging(log_file=log_file)

    # Create OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )

    def call_with_history(user_prompt: str = None) -> str:
        """Call the model with conversation history support"""
        if prompt_manager is None:
            raise ValueError("PromptManager is required for this function")
        
        # Get messages from PromptManager
        messages = prompt_manager.get_messages(user_prompt)
        
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                logging.info(f"Model call succeeded: model={model_name}")
                msg = response.choices[0].message
                content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else None)
                
                # Add assistant response to conversation history
                if content:
                    prompt_manager.add_assistant_message(content)
                
                return content

            except Exception as e:
                logging.error(f"Model call failed (attempt {attempt}): {e}")

                if attempt == max_retries:
                    raise

                time.sleep(retry_delay)

    return client, call_with_history


if __name__ == "__main__":
    # Test basic client
    client, call = create_openai_client(model_name="gemini-flash-latest-nothinking")
    result = call("Hello!")
    print("Basic client test:", result)
    
    # Test client with PromptManager
    pm = api_utils.PromptManager("You are a helpful assistant.")
    pm.add_user_message("What is AI?")
    
    client2, call_with_history = create_openai_client_with_prompt_manager(
        model_name="gemini-flash-latest-nothinking", 
        prompt_manager=pm
    )
    result2 = call_with_history("Tell me more about it.")
    print("Client with PromptManager test:", result2)
