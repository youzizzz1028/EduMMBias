# src/model_api/multimodal_openai.py
from openai import OpenAI
import logging
from . import api_utils

def create_openai_multimodal_client(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    log: str = None,
    max_retries: int = None,
    retry_delay: float = None,
    timeout: int = None,
):
    """
    Create an OpenAI client, supporting prompt + local image input
    """
    if not model_name:
        raise ValueError("model_name is required")
    default_cfg = api_utils.load_default_config()
    base_url = base_url or default_cfg.get("base_url")
    api_key = api_key or default_cfg.get("api_key")
    log_file = log or default_cfg.get("log", "llm_call.log")
    max_retries = max_retries or default_cfg.get("max_retries", 3)
    retry_delay = retry_delay or default_cfg.get("retry_delay", 1)
    timeout = timeout or default_cfg.get("timeout", 60)
    api_utils.setup_logging(log_file=log_file)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )
    def call_multimodal(prompt: str, image_paths: list, system_prompt: str = "You are a helpful assistant.") -> str:
        import mimetypes
        import base64
        import os
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        for img_path in image_paths:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type or not mime_type.startswith("image/"):
                raise ValueError(f"Not an image file: {img_path}")
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }
            })
        messages.append({"role": "user", "content": user_content})
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                logging.info(f"Multimodal model call succeeded: model={model_name}")
                msg = response.choices[0].message
                return getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else None)
            except Exception as e:
                logging.error(f"Multimodal model call failed (attempt {attempt}): {e}")
                if attempt == max_retries:
                    raise
                import time
                time.sleep(retry_delay)
        return None
    return client, call_multimodal
