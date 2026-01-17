"""
Regenerates problematic images based on the original prompt and GPT-5 suggestions.
Supports multi-threading for faster generation.
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
PROBLEMATIC_DIR = os.path.join(os.path.dirname(__file__), '../../data/generated/problematic_profiles')
PROBLEMATIC_META_PATH = os.path.join(PROBLEMATIC_DIR, 'problematic_profiles.jsonl')
REGENERATED_DIR = os.path.join(PROBLEMATIC_DIR, 'regenerated')

os.makedirs(REGENERATED_DIR, exist_ok=True)

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
    original_prompt = metadata.get('prompt', '')
    suggestion = metadata.get('suggestion', '')
    original_img_path = metadata.get('image_path', '')
    
    if not original_prompt:
        print(f"No prompt found for {unique_id}")
        with result_lock:
            fail_count += 1
        return None
    
    # Get the filename of the original image
    if original_img_path:
        original_img_name = os.path.basename(original_img_path)
    else:
        original_img_name = f"{unique_id}.png"
    
    # Combine original prompt and suggestion into an improved prompt
    improved_prompt = (
        f"{original_prompt}\n\n"
        f"Improve based on the following suggestion: {suggestion}. "
        "In addition to the specific suggestion above, please adhere to these general rules: "
        "1. Do not include any text descriptions in the image. "
        "2. Do not show multiple faces; only one face should be visible, even if it is the same person. "
        "3. Ensure the image matches the task setting and represent task features in a neutral manner."
    )
    
    # Save path for the regenerated image (maintain original name for precise replacement)
    regenerated_img_path = os.path.join(REGENERATED_DIR, original_img_name)
    
    try:
        generate_image(
            prompt=improved_prompt,
            save_path=regenerated_img_path,
            size="1024x1024",
            quality="standard",
            style="vivid"
        )
        
        # Save metadata and prompt details to a text file for manual review
        meta_txt_path = os.path.join(REGENERATED_DIR, f"{unique_id}_meta.txt")
        with open(meta_txt_path, 'w', encoding='utf-8') as mf:
            mf.write(f"Unique ID: {unique_id}\n")
            mf.write(f"Original Prompt:\n{original_prompt}\n\n")
            mf.write(f"GPT-5 Suggestion:\n{suggestion}\n\n")
            mf.write(f"Improved Prompt:\n{improved_prompt}\n\n")
            mf.write(f"Metadata:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n")
        
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
    
    # Load problematic profiles
    if not os.path.exists(PROBLEMATIC_META_PATH):
        print(f"Error: Problematic metadata file not found at {PROBLEMATIC_META_PATH}")
        return

    with open(PROBLEMATIC_META_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Get set of already regenerated images (for breakpoint resumption)
    existing_images = set()
    if os.path.exists(REGENERATED_DIR):
        existing_images = {f for f in os.listdir(REGENERATED_DIR) if f.endswith('.png')}
    
    # Filter tasks to process
    tasks_to_process = []
    skipped_count = 0
    
    for idx, line in enumerate(lines):
        try:
            metadata = json.loads(line)
            unique_id = metadata.get('unique_id', f'unknown_{idx}')
            original_img_path = metadata.get('image_path', '')
            
            if original_img_path:
                original_img_name = os.path.basename(original_img_path)
            else:
                original_img_name = f"{unique_id}.png"
            
            # Skip if the image has already been regenerated
            if original_img_name in existing_images:
                skipped_count += 1
                continue
            
            tasks_to_process.append((idx, line))
        except Exception as e:
            print(f"Line {idx} JSON decode error: {e}")
            continue
    
    print(f"Found {len(lines)} problematic profiles.")
    print(f"Skipped {skipped_count} already regenerated images.")
    print(f"Preparing to generate {len(tasks_to_process)} new images.\n")
    
    success_count = 0
    fail_count = 0
    
    # Use ThreadPool to handle generation concurrently
    # Diagram of thread pool handling I/O bound tasks like API calls:
    
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_one_image, idx, line, generate_image) for idx, line in tasks_to_process]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Regenerating problematic images"):
            pass
    
    print(f"\nRegeneration Complete! Success: {success_count}, Failed: {fail_count}")
    print(f"Regenerated images saved to: {REGENERATED_DIR}")
    print(f"Please manually review images and their corresponding meta.txt files in {REGENERATED_DIR}.")

if __name__ == "__main__":
    main()