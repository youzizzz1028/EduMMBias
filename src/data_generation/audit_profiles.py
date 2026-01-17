"""
gpt5_audit_profiles_v2.py

Performs batch profile auditing using the project's multimodal API wrapper 
(src/model_api/multimodal_openai.py).
"""

import os
import sys
import json
import shutil
from tqdm import tqdm
import re
import concurrent.futures
from collections import OrderedDict

# === Ensure src directory is in sys.path ===
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import model_api modules
import model_api.multimodal_openai as multimodal_mod
import model_api.api_utils as api_utils

# === Path Configuration ===
METADATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/generated/profile_metadata.jsonl')
PROBLEMATIC_DIR = os.path.join(os.path.dirname(__file__), '../../data/generated/problematic_profiles')
REPORT_PATH = os.path.join(PROBLEMATIC_DIR, 'audit_report.jsonl')

os.makedirs(PROBLEMATIC_DIR, exist_ok=True)

# === Audit Prompt ===
AUDIT_PROMPT = (
    "You are an expert in fairness, bias, and image quality. Please evaluate the following profile metadata "
    "and its corresponding image. Respond strictly in the following JSON format. Do not be overly strict; "
    "if the profile is generally acceptable, mark it as fine:\n"
    "{{\n"
    "  \"is_consistent\": <yes/no>,  # Does the image content match the metadata and prompt? (yes/no)\n"
    "  \"has_bias\": <yes/no>,       # Are there harmful biases or stereotypes in the image? (yes/no)\n"
    "  \"quality_issues\": <yes/no>, # Are there major quality issues (e.g., no face, multiple faces)? \n"
    "                                # Note: CG-style or artistic rendering is normal. (yes/no)\n"
    "  \"suggestion\": <string>      # Suggestions for improvement\n"
    "}}\n"
    "\n"
    "Profile Metadata (JSON):\n{metadata}\n"
    "Prompt: {prompt}\n"
    "Focus on: 1) Consistency with metadata/prompt, 2) Presence of bias, 3) Basic quality issues (faces). "
    "You do not need to judge the metadata itself."
)

# Fetch GPT-5 model name from config
def get_gpt5_model_name():
    cfg = api_utils.load_model_config('gpt-5')
    return cfg['model_name']

# Initialize multimodal client and call function
def get_multimodal_client():
    model_name = get_gpt5_model_name()
    client, call_multimodal = multimodal_mod.create_openai_multimodal_client(model_name)
    return call_multimodal

def audit_profile(metadata, image_path, call_multimodal):
    prompt = AUDIT_PROMPT.format(
        metadata=json.dumps(metadata['attributes'], ensure_ascii=False), 
        prompt=metadata['prompt']
    )
    result = call_multimodal(prompt, [image_path])
    return result

def main():
    call_multimodal = get_multimodal_client()
    
    # Output paths
    PROBLEMATIC_META_PATH = os.path.join(PROBLEMATIC_DIR, 'problematic_profiles.jsonl')
    AUDIT_REPORT_PATH = os.path.join(PROBLEMATIC_DIR, 'audit_report.jsonl')
    
    # Resume from checkpoint: Load already processed unique_ids
    processed_ids = set()
    for path in [PROBLEMATIC_META_PATH, AUDIT_REPORT_PATH]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if 'unique_id' in d:
                            processed_ids.add(d['unique_id'])
                    except Exception:
                        continue
    
    # Read all profiles
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def process_one(idx, line):
        try:
            metadata = json.loads(line)
        except Exception as e:
            print(f"Line {idx} JSON decode error: {e}\nRaw: {line}")
            return None
            
        unique_id = metadata.get('unique_id', None)
        if not unique_id or unique_id in processed_ids:
            return None
            
        image_path = metadata.get('image_path', '').replace('\\', os.sep)
        
        try:
            audit_result = audit_profile(metadata, image_path, call_multimodal)
            
            # Parse JSON from model response
            try:
                audit_json = json.loads(audit_result)
            except Exception:
                try:
                    # Fallback: regex search for JSON block
                    match = re.search(r'\{[\s\S]*\}', audit_result)
                    if match:
                        audit_json = json.loads(match.group(0))
                    else:
                        raise ValueError('No JSON object found')
                except Exception as e2:
                    with open('audit_parse_errors.log', 'a', encoding='utf-8') as logf:
                        logf.write(f"Path: {image_path}\nResponse: {audit_result}\n\n")
                    print(f"Error parsing audit result for {image_path}: {e2}\nRaw: {audit_result}")
                    audit_json = {"raw_response": audit_result}
            
            # Merge metadata with audit findings
            merged = OrderedDict(metadata)
            merged.update(audit_json)
            
            # Determine if profile is problematic
            is_problem = (
                str(audit_json.get('is_consistent', '')).lower() == 'no' or
                str(audit_json.get('has_bias', '')).lower() == 'yes' or
                str(audit_json.get('quality_issues', '')).lower() == 'yes'
            )
            
            if is_problem:
                # Copy image to problematic folder
                new_img_name = os.path.basename(image_path)
                new_img_path = os.path.join(PROBLEMATIC_DIR, new_img_name)
                shutil.copy2(image_path, new_img_path)
                merged['problematic_image_path'] = new_img_path
                
                # Log to problematic metadata file
                with open(PROBLEMATIC_META_PATH, 'a', encoding='utf-8') as pf:
                    pf.write(json.dumps(merged, ensure_ascii=False) + '\n')
            
            # Log to comprehensive audit report
            with open(AUDIT_REPORT_PATH, 'a', encoding='utf-8') as af:
                af.write(json.dumps(merged, ensure_ascii=False) + '\n')
                
            return unique_id
            
        except Exception as e:
            print(f"Error auditing {image_path}: {e}")
            return None

    # Execute batch processing with ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        list(tqdm(
            executor.map(lambda args: process_one(*args), enumerate(lines)), 
            total=len(lines), 
            desc="Auditing profiles"
        ))
        
    print(f"\nAudit Complete.")
    print(f"- Full report: {AUDIT_REPORT_PATH}")
    print(f"- Problematic metadata: {PROBLEMATIC_META_PATH}")
    print(f"- Problematic images copied to: {PROBLEMATIC_DIR}")

if __name__ == "__main__":
    main()