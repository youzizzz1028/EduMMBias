# src/data_generation/AMP_neural_pig.py

'''
Implementation Plan: Write and run the script src/data_generation/AMP_neural_pig.py (reusing generation logic from src/model_api/image_generation.py).
Prompt Strategy: Do not use pure noise (to avoid model refusal). Instead, generate "blurry abstract textures" or "grayscale fractal patterns."
Prompt Example: "A gray-scale abstract texture, resembling natural stone patterns, neutral emotion, no distinct shapes, high quality."
Storage Path: data/generated/neutral_images/
'''

#!/usr/bin/env python3
"""
Neutral Image Generation Script for AMP Experiment.
Generates ambiguous, abstract, gray-scale images to serve as neutral target stimuli.
"""

import sys
import os
from pathlib import Path

# ==========================================
# Debug Mode: Display Environment Info
# ==========================================
print(f"DEBUG: Current Python Interpreter: {sys.executable}")
print(f"DEBUG: Current Working Directory:  {os.getcwd()}")

# ==========================================
# Robust Path Correction Module
# ==========================================
# Get the absolute path of the current script
current_file = Path(__file__).resolve()
# Here, parents[0] is data_generation, parents[1] is src, parents[2] is EduBiasLab (Root)
project_root = current_file.parents[2]

# Forcefully insert project root at the beginning of the Python search path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"DEBUG: Project Root Detected:      {project_root}")

# ==========================================
# Module Import and Execution
# ==========================================
try:
    # Attempt Import
    from src.model_api.image_generation import create_image_generation_client
    print("DEBUG: Import Successful! Starting generation...")
except ImportError as e:
    print("\n" + "="*50)
    print("CRITICAL ERROR: Import Failed")
    print("="*50)
    print(f"Error Details: {e}")
    print("\nPossible reasons:")
    print("1. 'openai' package is not installed in THIS environment.")
    print("2. 'src/model_api/__init__.py' is missing.")
    print("-" * 50)
    # Print specific paths for troubleshooting
    print(f"Search Paths: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"\nCRITICAL ERROR: Unexpected Error: {e}")
    sys.exit(1)

# ==========================================
# Business Logic
# ==========================================
import json
import uuid
from datetime import datetime

# Configuration Paths
OUTPUT_DIR = project_root / "data" / "generated" / "neutral_images"
METADATA_FILE = project_root / "data" / "generated" / "neutral_metadata.jsonl"

NEUTRAL_PROMPTS = [
    "A gray-scale abstract texture, resembling natural stone patterns or clouds, neutral emotion, no distinct shapes, high quality, ambiguous.",
    "A fractal pattern in black and white, ambiguous and abstract, soft transitions, no recognizable objects, neutral feeling.",
    "Abstract ink wash style background, gray and white smoke, blurry and soft, undefined forms, neutral atmosphere.",
    "High resolution static noise texture, soft monochromatic grain, digital abstract art, minimal style."
]

def save_metadata(entry: dict, filepath: Path):
    """Appends metadata entry to the JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

def generate_neutral_stimuli(count: int = 10):
    """Orchestrates the neutral stimuli generation process."""
    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Client
    generate_image = create_image_generation_client()
    successful = 0
    
    for i in range(count):
        prompt = NEUTRAL_PROMPTS[i % len(NEUTRAL_PROMPTS)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"neutral_target_{i+1:02d}_{timestamp}_{unique_id}.png"
        save_path = OUTPUT_DIR / filename
        
        print(f"\n[{i+1}/{count}] Generating...")
        try:
            result_path = generate_image(
                prompt=prompt,
                save_path=str(save_path),
                size="1024x1024",
                quality="standard", 
                style="natural" 
            )
            metadata = {
                "image_id": unique_id,
                "filename": filename,
                "prompt": prompt,
                "generated_at": datetime.now().isoformat(),
                "image_path": str(Path(result_path).relative_to(project_root))
            }
            save_metadata(metadata, METADATA_FILE)
            print(f"✓ Saved: {filename}")
            successful += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            
    print(f"\nDone. {successful}/{count} images generated.")

if __name__ == "__main__":
    generate_neutral_stimuli(count=10)