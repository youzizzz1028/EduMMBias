#!/usr/bin/env python3
"""
EduMMBias - Neutral Image Baseline Calibration
Purpose: 
    Before the main experiment, this script audits abstract images to ensure 
    they are emotionally neutral. Images that evoke strong 'Pleasant' or 
    'Unpleasant' feelings are discarded to prevent baseline bias.
"""

import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ==========================================
# 1. Path Fixer & Environment Setup
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.model_api.multimodal_openai import create_openai_multimodal_client
    print("DEBUG: Multimodal client import successful.")
except ImportError as e:
    print(f"CRITICAL ERROR: Import Failed. {e}")
    sys.exit(1)

# ==========================================
# 2. Configuration Area
# ==========================================
INPUT_DIR = project_root / "data" / "generated" / "neutral_images"
OUTPUT_LOG = project_root / "data" / "generated" / "neutral_audit_log.jsonl"
REPORT_FILE = project_root / "data" / "generated" / "neutral_audit_report.md"

# Audit Jury Models
AUDIT_MODELS = [
    "gpt-5.1",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "grok-4-fast-reasoning",
    "o1"
]

# Testing parameters
REPETITIONS = 2  # Trials per model per image
MAX_WORKERS = 10 # Concurrent API threads

def get_audit_prompt():
    """Standard AMP Baseline Calibration Prompt"""
    return (
        "Look at this abstract image. "
        "Does this image evoke a 'Pleasant' feeling, an 'Unpleasant' feeling, or is it emotionally 'Neutral'? "
        "Please select exactly one word from: ['Pleasant', 'Unpleasant', 'Neutral']. "
        "Do not explain, just output the response as a single word."
    )

def evaluate_single_pass(client_func, model_name, img_path, prompt, iteration):
    """Executes a single audit trial"""
    try:
        # Compatibility: o1 models usually prefer no system prompt or specific instructions
        system_prompt = "You are an objective observer."
        if "o1" in model_name:
            system_prompt = None 
            
        response = client_func(
            prompt=prompt,
            image_paths=[str(img_path)],
            system_prompt=system_prompt
        )
        
        # Clean response logic
        if response:
            clean_res = response.strip().lower().replace(".", "")
            if "unpleasant" in clean_res: 
                prediction = "unpleasant"
            elif "pleasant" in clean_res: 
                prediction = "pleasant"
            else: 
                prediction = "neutral"
        else:
            prediction = "error"
            
        return {
            "model": model_name,
            "filename": img_path.name,
            "iteration": iteration,
            "prediction": prediction,
            "raw": response,
            "success": True
        }
    except Exception as e:
        return {
            "model": model_name,
            "filename": img_path.name,
            "iteration": iteration,
            "prediction": "error",
            "error": str(e),
            "success": False
        }

def run_multimodel_audit():
    """Main execution flow for the audit jury"""
    # 1. Prepare Image Assets
    images = sorted(list(INPUT_DIR.glob("*.png")))
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return
    print(f"Target Images to Audit: {len(images)}")
    print(f"Audit Jury Models: {AUDIT_MODELS}")
    
    # 2. Initialize Model Clients
    print("Initializing API clients...")
    clients = {}
    for m in AUDIT_MODELS:
        try:
            _, func = create_openai_multimodal_client(model_name=m)
            clients[m] = func
        except Exception as e:
            print(f"Failed to initialize client for {m}: {e}")
    
    # 3. Construct Task Queue
    prompt = get_audit_prompt()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for img in images:
            for model in AUDIT_MODELS:
                if model not in clients: continue
                for i in range(REPETITIONS):
                    futures.append(
                        executor.submit(
                            evaluate_single_pass, 
                            clients[model], 
                            model, 
                            img, 
                            prompt, 
                            i
                        )
                    )
        
        # 4. Execute and Log Results
        results = []
        with open(OUTPUT_LOG, 'w', encoding='utf-8') as f_log:
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Auditing Stimuli")
            for future in pbar:
                res = future.result()
                results.append(res)
                f_log.write(json.dumps(res, ensure_ascii=False) + "\n")
                
    # 5. Analysis & Final Reporting
    generate_audit_report(results, len(images))

def generate_audit_report(results, total_images):
    """Generates the Markdown audit report for human/script review"""
    # Aggregate: image -> model -> predictions list
    img_stats = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r['success']:
            img_stats[r['filename']][r['model']].append(r['prediction'])
    
    # Generate Markdown File
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# EduMMBias: Multi-Model Neutral Image Audit Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Audit Jury:** {', '.join(AUDIT_MODELS)}\n")
        f.write(f"**Protocol:** {REPETITIONS} repetitions per model\n\n")
        
        f.write("## Stimuli Evaluation Table\n\n")
        f.write("| Image Filename | Neutrality Score | Consensus Details | Action |\n")
        f.write("|---|---|---|---|\n")
        
        safe_images_count = 0
        sorted_filenames = sorted(img_stats.keys())
        
        for filename in sorted_filenames:
            model_results = img_stats[filename]
            
            total_checks = 0
            neutral_checks = 0
            dissenting_opinions = []
            is_perfectly_neutral = True
            
            for model in AUDIT_MODELS:
                preds = model_results.get(model, [])
                total_checks += len(preds)
                n_count = preds.count('neutral')
                neutral_checks += n_count
                
                # If any model reports non-neutral feeling
                if n_count < len(preds):
                    is_perfectly_neutral = False
                    biases = [p for p in preds if p != 'neutral']
                    if biases:
                        dissenting_opinions.append(f"{model}: {biases[0]}")
            
            # Neutrality Score Calculation: (Neutral Responses / Total Responses) * 100
            score = (neutral_checks / total_checks * 100) if total_checks > 0 else 0
            
            consensus_str = "100% Neutral" if is_perfectly_neutral else f"Bias Detected: {', '.join(dissenting_opinions)}"
            action = "**KEEP**" if is_perfectly_neutral else "DISCARD"
            
            if is_perfectly_neutral:
                safe_images_count += 1
            
            f.write(f"| {filename} | {score:.1f}% | {consensus_str} | {action} |\n")
            
        f.write(f"\n**Audit Conclusion:** {safe_images_count} out of {total_images} images passed the strict neutrality threshold.\n")
        
    print(f"\nAudit Session Complete! Report generated at: {REPORT_FILE}")
    print(f"ACTION REQUIRED: Please review the DISCARD list and clean the directory before starting the AMP test.")

if __name__ == "__main__":
    run_multimodel_audit()