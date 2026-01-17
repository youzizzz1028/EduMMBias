"""
Replacement Confirmation Script: Replaces problematic images in the original 'profile_images' 
folder with newly generated images from the 'regenerated' folder.

Please ensure you have manually reviewed the new images in the 'regenerated' folder before running this.
"""

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

# Path Configuration
PROBLEMATIC_DIR = os.path.join(os.path.dirname(__file__), '../../data/generated/problematic_profiles')
REGENERATED_DIR = os.path.join(PROBLEMATIC_DIR, 'regenerated')
PROBLEMATIC_META_PATH = os.path.join(PROBLEMATIC_DIR, 'problematic_profiles.jsonl')

def main():
    print("=== Image Replacement Confirmation Script ===")
    print(f"Reading from: {REGENERATED_DIR}")
    print(f"Targeting original paths defined in the metadata 'image_path' field.\n")
    
    # Read problematic profile metadata to find replacement targets
    replacements = []
    if not os.path.exists(PROBLEMATIC_META_PATH):
        print(f"Error: Metadata file not found at {PROBLEMATIC_META_PATH}")
        return

    with open(PROBLEMATIC_META_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                metadata = json.loads(line)
                original_img_path = metadata.get('image_path', '')
                original_img_name = os.path.basename(original_img_path)
                
                # The regenerated image should have the same filename as the original
                regenerated_img_path = os.path.join(REGENERATED_DIR, original_img_name)
                
                if os.path.exists(regenerated_img_path):
                    replacements.append({
                        'unique_id': metadata.get('unique_id', ''),
                        'original_path': original_img_path,
                        'regenerated_path': regenerated_img_path,
                        'original_name': original_img_name
                    })
            except Exception:
                continue
    
    if not replacements:
        print("No replaceable images found! Ensure filenames in 'regenerated' match the original filenames.")
        return
    
    print(f"Found {len(replacements)} regenerated images ready for replacement:\n")
    for item in replacements:
        print(f"  ID: {item['unique_id']} | File: {item['original_name']}")
    
    # User Confirmation
    confirm = input("\n\nConfirm replacement? This will overwrite original files. (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Replacement cancelled.")
        return
    
    # Execute Replacement Logic
    success_count = 0
    fail_count = 0
    
    
    
    for item in tqdm(replacements, desc="Replacing images"):
        try:
            original_path = item['original_path']
            regenerated_path = item['regenerated_path']
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            
            # Backup the original file (append .backup suffix)
            if os.path.exists(original_path):
                backup_path = original_path + '.backup'
                shutil.copy2(original_path, backup_path)
            
            # Overwrite the original file with the new regenerated image
            shutil.copy2(regenerated_path, original_path)
            success_count += 1
            
        except Exception as e:
            print(f"Failed to replace {item['unique_id']}: {e}")
            fail_count += 1
    
    print(f"\nReplacement Process Complete!")
    print(f"Successfully Replaced: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Original files have been backed up with the '.backup' extension.")

if __name__ == "__main__":
    main()