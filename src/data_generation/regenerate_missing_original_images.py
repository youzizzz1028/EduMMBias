"""
Detects and regenerates missing original images in profile_images/
Supports multi-threading for faster generation and breakpoint resumption.
"""

import os
import sys
import json
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure src directory is in sys.path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model_api.image_generation import create_image_generation_client

# Path Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/generated')
PROFILE_META_PATH = os.path.join(DATA_DIR, 'profile_metadata.jsonl')
PROFILE_IMAGES_DIR = os.path.join(DATA_DIR, 'profile_images')

os.makedirs(PROFILE_IMAGES_DIR, exist_ok=True)

# Thread-safe counters
result_lock = threading.Lock()
success_count = 0
fail_count = 0

def process_one_image(idx, line, generate_image):
    global success_count, fail_count
    
    try:
        metadata = json.loads(line)
    except Exception as e:
        print(f"Line {idx} JSON decode error: {e}")
        with result_lock:
            fail_count += 1
        return None
    
    unique_id = metadata.get('unique_id', f'unknown_{idx}')
    prompt = metadata.get('prompt', '')
    image_path = metadata.get('image_path', '')
    
    if not prompt:
        print(f"No prompt found for {unique_id}")
        with result_lock:
            fail_count += 1
        return None
    
    # Extract image filename
    if image_path:
        image_name = os.path.basename(image_path)
    else:
        image_name = f"{unique_id}.png"
    
    # Full destination path
    full_image_path = os.path.join(PROFILE_IMAGES_DIR, image_name)
    
    # Check if image already exists (Breakpoint resumption)
    if os.path.exists(full_image_path):
        with result_lock:
            success_count += 1
        return unique_id
    
    # Image missing, initiate generation
    try:
        # Appended English rules for consistency with an English VLM pipeline
        refined_prompt = (
            f"{prompt} "
            "Follow these general rules: "
            "1. Do not include any text descriptions in the image. "
            "2. Do not show multiple faces; only one face should be visible, even if it is the same person. "
            "3. Ensure the image matches the task setting, and represent task features in a neutral manner."
        )
        
        generate_image(
            prompt=refined_prompt,
            save_path=full_image_path,
            size="1024x1024",
            quality="standard",
            style="vivid"
        )
        
        with result_lock:
            success_count += 1
        return unique_id
        
    except Exception as e:
        print(f"✗ Failed to regenerate {unique_id}: {e}")
        with result_lock:
            fail_count += 1
        return None

def main():
    global success_count, fail_count
    
    # Initialize image generation client
    generate_image = create_image_generation_client()
    
    # Load original metadata
    if not os.path.exists(PROFILE_META_PATH):
        print(f"Error: Metadata file not found at {PROFILE_META_PATH}")
        return

    with open(PROFILE_META_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Get set of already existing image filenames
    existing_images = set()
    if os.path.exists(PROFILE_IMAGES_DIR):
        existing_images = {f for f in os.listdir(PROFILE_IMAGES_DIR) if f.endswith('.png')}
    
    # Filter for missing images
    tasks_to_process = []
    missing_count = 0
    
    for idx, line in enumerate(lines):
        try:
            metadata = json.loads(line)
            unique_id = metadata.get('unique_id', f'unknown_{idx}')
            image_path = metadata.get('image_path', '')
            
            if image_path:
                image_name = os.path.basename(image_path)
            else:
                image_name = f"{unique_id}.png"
            
            # Check if image is missing
            if image_name not in existing_images:
                missing_count += 1
                tasks_to_process.append((idx, line))
        except Exception as e:
            print(f"Line {idx} JSON decode error: {e}")
            continue
    
    print(f"Total profiles to process: {len(lines)}")
    print(f"Existing images: {len(existing_images)}")
    print(f"Missing images to regenerate: {missing_count}\n")
    
    if missing_count == 0:
        print("✓ All images are present. No regeneration needed.")
        return
    
    success_count = 0
    fail_count = 0
    
    # Use ThreadPool to handle generation concurrently
    # Diagram of thread pool handling I/O bound tasks like API calls:
    
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_one_image, idx, line, generate_image) for idx, line in tasks_to_process]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Regenerating missing original images"):
            pass
    
    print(f"\nRegeneration Complete! Success: {success_count}, Failed: {fail_count}")
    print(f"Regenerated images saved to: {PROFILE_IMAGES_DIR}")

if __name__ == '__main__':
    main()