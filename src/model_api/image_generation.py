# src/model_api/image_generation.py
# api_base_url和api_key通过api_utils.py中的load_default_config函数加载默认配置
from openai import OpenAI
from . import api_utils
import requests
import logging
import time
from pathlib import Path
from typing import Optional

model_name = "dall-e-3"


def generate_image(
    prompt: str,
    save_path: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    base_url: str = None,
    api_key: str = None,
    max_retries: int = None,
    retry_delay: float = None,
    timeout: int = None,
) -> str:
    """
    Generate image using DALL-E 3 and save to local
    
    Args:
        prompt: Image generation prompt
        save_path: Path to save the image
        size: Image size, options: "1024x1024", "1024x1792", "1792x1024"
        quality: Image quality, options: "standard", "hd"
        style: Image style, options: "vivid", "natural"
        base_url: API base URL, use default config if None
        api_key: API key, use default config if None
        max_retries: Maximum retry times
        retry_delay: Retry delay in seconds
        timeout: Timeout in seconds
    
    Returns:
        str: Saved image file path
    
    Raises:
        ValueError: Invalid arguments
        Exception: API call failed
    """
    # Validate parameters
    if not prompt:
        raise ValueError("prompt is required")
    if not save_path:
        raise ValueError("save_path is required")
    
    # Validate size parameter
    valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
    if size not in valid_sizes:
        raise ValueError(f"size must be one of {valid_sizes}")
    
    # Validate quality parameter
    valid_qualities = ["standard", "hd"]
    if quality not in valid_qualities:
        raise ValueError(f"quality must be one of {valid_qualities}")
    
    # Validate style parameter
    valid_styles = ["vivid", "natural"]
    if style not in valid_styles:
        raise ValueError(f"style must be one of {valid_styles}")
    
    # Load default config
    default_cfg = api_utils.load_default_config()
    
    # Set parameters, use provided value or default
    base_url = base_url or default_cfg.get("base_url")
    api_key = api_key or default_cfg.get("api_key")
    max_retries = max_retries or default_cfg.get("max_retries", 3)
    retry_delay = retry_delay or default_cfg.get("retry_delay", 1)
    timeout = timeout or default_cfg.get("timeout", 60)
    
    # Setup logging
    log_file = default_cfg.get("log", "llm_call.log")
    api_utils.setup_logging(log_file=log_file)
    
    # Create OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )
    
    # Ensure save directory exists
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Retry logic
    for attempt in range(max_retries + 1):
        try:
            logging.info(f"Generating image with prompt: {prompt}")
            
            # Call DALL-E 3 API to generate image
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1  # Generate only one image
            )
            
            # Get image URL
            image_url = response.data[0].url
            logging.info(f"Image generated successfully: {image_url}")
            
            # Download image
            image_response = requests.get(image_url, timeout=timeout)
            image_response.raise_for_status()
            
            # Save image to local
            with open(save_path, 'wb') as f:
                f.write(image_response.content)
            
            logging.info(f"Image saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Image generation failed (attempt {attempt}): {e}")
            
            if attempt == max_retries:
                raise
            
            time.sleep(retry_delay)
    
    return save_path


def create_image_generation_client(
    base_url: str = None,
    api_key: str = None,
    max_retries: int = None,
    retry_delay: float = None,
    timeout: int = None,
):
    """
    Create image generation client, return configured generation function
    
    Args:
        base_url: API base URL, use default config if None
        api_key: API key, use default config if None
        max_retries: Maximum retry times
        retry_delay: Retry delay in seconds
        timeout: Timeout in seconds
    
    Returns:
        function: Configured image generation function
    """
    # Load default config
    default_cfg = api_utils.load_default_config()
    
    # Set parameters
    base_url = base_url or default_cfg.get("base_url")
    api_key = api_key or default_cfg.get("api_key")
    max_retries = max_retries or default_cfg.get("max_retries", 3)
    retry_delay = retry_delay or default_cfg.get("retry_delay", 1)
    timeout = timeout or default_cfg.get("timeout", 60)
    
    def configured_generate_image(
        prompt: str,
        save_path: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
    ) -> str:
        """Generate image with pre-configured parameters"""
        return generate_image(
            prompt=prompt,
            save_path=save_path,
            size=size,
            quality=quality,
            style=style,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )
    
    return configured_generate_image


if __name__ == "__main__":
    # Test image generation function
    try:
        # Create client with default config
        generate_image_func = create_image_generation_client()
        
        # Test generating image
        test_prompt = "A beautiful sunset over the mountains"
        test_save_path = "test_generated_image.png"
        
        result = generate_image_func(
            prompt=test_prompt,
            save_path=test_save_path
        )
        
        print(f"Image generated successfully: {result}")
        
    except Exception as e:
        print(f"Image generation test failed: {e}")
