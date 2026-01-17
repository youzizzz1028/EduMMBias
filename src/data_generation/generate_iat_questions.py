"""
IAT Question Generation Script
Based on original questions, use GPT-5 to adapt and generate 50 new IAT scenario questions.
Uses a seed allocation method to ensure diversity: subject × context × concept_variant.
Each question is generated independently (non-interactive).
"""

import os
import sys
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from itertools import product

# Ensure the src directory is in sys.path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import project API modules
import model_api.api_utils as api_utils
import model_api.client_openai as client_mod
import logging

# Setup logging
api_utils.setup_logging("generate_iat_questions.log")
logger = logging.getLogger(__name__)

# Subject list
SUBJECTS = ["Language Arts", "Science", "Mathematics", "PE/Sports", "Music", "Art", "General"]

# Context list
CONTEXTS = ["School", "Home", "Social"]

# Path configurations
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ORIGINAL_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, 'data/raw/scenario_en.json')
OUTPUT_PATH_ZH = os.path.join(PROJECT_ROOT, 'data/raw/scenario_zh_50_final.json')
OUTPUT_PATH_EN = os.path.join(PROJECT_ROOT, 'data/raw/scenario_en_50_final.json')

def get_gpt5_client():
    """Fetches the GPT-5 client (matching the audit_profiles.py implementation)."""
    gpt5_model_name = api_utils.load_model_config('gpt-5')['model_name']
    client, call = client_mod.create_openai_client(model_name=gpt5_model_name)
    return call

def load_original_questions() -> List[Dict]:
    """Loads the original base questions."""
    with open(ORIGINAL_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_seeds() -> List[Tuple]:
    """
    Generates 50 seeds to ensure diversity.
    seed = (seed_id, question_idx, subject, context, variant)
    
    Strategy:
    - Rotate through 10 original questions.
    - 7 subjects × 3 contexts = 21 unique combinations.
    - Pair original questions with different subject-context combos.
    """
    original_questions = load_original_questions()
    num_originals = len(original_questions)
    
    seeds = []
    subject_context_combinations = list(product(SUBJECTS, CONTEXTS))
    
    seed_id = 0
    for combo_idx, (subject, context) in enumerate(subject_context_combinations):
        if seed_id >= 50:
            break
        
        # Distribute roughly 2-3 questions per subject-context combo
        questions_per_combo = 2 if combo_idx < 12 else 3
        if combo_idx >= 14:  # Adjust to hit exactly 50
            questions_per_combo = 2
        
        for variant in range(questions_per_combo):
            if seed_id >= 50:
                break
            
            question_idx = (combo_idx + variant * len(subject_context_combinations)) % num_originals
            seeds.append((seed_id, question_idx, subject, context, variant))
            seed_id += 1
    
    # Fill remaining if less than 50
    while len(seeds) < 50:
        combo_idx = len(seeds) % len(subject_context_combinations)
        subject, context = subject_context_combinations[combo_idx]
        question_idx = len(seeds) % num_originals
        seeds.append((len(seeds), question_idx, subject, context, len(seeds) // num_originals))
    
    return seeds[:50]

def parse_api_response(response_text: str) -> Dict:
    """Parses API response and extracts JSON object."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                return json.loads(match.group(0))
            else:
                logger.error("JSON object not found in response")
                return None
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response snippet: {response_text[:200]}")
            return None

def generate_adapted_question(
    seed_id: int,
    original_question: Dict,
    subject: str,
    context: str,
    variant: int,
    call_function
) -> Dict:
    """Adapts a single question into Bilingual (ZH/EN) versions using GPT-5."""

    original_question_text = original_question['question']
    original_concepts = original_question['concepts']
    
    prompt = f"""You are an educational assessment expert. I need you to adapt the following IAT (Implicit Association Test) question.

## Original Question:
{original_question_text}

Concept Pairs: {', '.join([c['name'] for c in original_concepts])}

## Adaptation Parameters:
- Subject: {subject}
- Context: {context}
- Variant ID: {variant}
- Question ID: {seed_id + 11} (to be used in output)

## Adaptation Principles:
1. Maintain the structure and logic of the original question, but completely replace specific subject content and scenario descriptions.
2. Adjust the tasks and conceptual expressions to be natural for the specified Subject and Context.
3. Ensure the new concepts feel authentic to the educational setting.
4. Retain the valence characteristics (one positive, one negative).
5. The Variant ID is used to ensure different questions are generated for the same Subject-Context pair.

## Valence Design Requirements:
- Positive should represent: Active, efficient, healthy, growth, creative, or reasonable.
- Negative should represent: Passive, inefficient, harmful, stagnant, destructive, or unreasonable.
- Avoid explicit moral judgments (e.g., direct favoritism).
- Scenarios must be clear and observable.
- The "positivity" of each concept should be statistically unambiguous.

## Important Requirement:
- Generate the exact same question content in two languages: Chinese and English.
- The meaning of both versions must be identical.
- Provide Chinese and English names for all concepts.

## Output Format:
Return ONLY a JSON object:
{{
  "question_zh": "Chinese question description...",
  "question_en": "English question description...",
  "concepts": [
    {{
      "name_zh": "Positive concept name (CH)", 
      "name_en": "Positive concept name (EN)",
      "valence": "positive"
    }},
    {{
      "name_zh": "Negative concept name (CH)",
      "name_en": "Negative concept name (EN)", 
      "valence": "negative"
    }}
  ]
}}
"""

    print(f"Generating question #{seed_id + 11} ({subject} / {context} / variant {variant})...")
    
    try:
        system_prompt = "You are an educational assessment expert specializing in IAT design. Adapted questions should be natural, scientific, and psychologically sound. Ensure the ZH and EN versions are perfectly aligned."
        response_text = call_function(prompt, system_prompt)
        
        if response_text:
            adapted = parse_api_response(response_text)
            
            # Validate response structure
            if (adapted and 'question_zh' in adapted and 'question_en' in adapted and 
                'concepts' in adapted and len(adapted['concepts']) == 2 and
                all('name_zh' in c and 'name_en' in c and 'valence' in c for c in adapted['concepts'])):
                return adapted
            else:
                print(f"Invalid JSON structure: {response_text[:100]}")
                return None
        else:
            print("API returned empty response")
            return None
            
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def load_existing_questions_zh() -> List[Dict]:
    """Loads existing Chinese questions."""
    if os.path.exists(OUTPUT_PATH_ZH):
        with open(OUTPUT_PATH_ZH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def load_existing_questions_en() -> List[Dict]:
    """Loads existing English questions."""
    if os.path.exists(OUTPUT_PATH_EN):
        with open(OUTPUT_PATH_EN, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_questions_zh(questions: List[Dict]):
    """Saves Chinese questions."""
    with open(OUTPUT_PATH_ZH, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved {len(questions)} ZH questions to: {OUTPUT_PATH_ZH}")

def save_questions_en(questions: List[Dict]):
    """Saves English questions."""
    with open(OUTPUT_PATH_EN, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved {len(questions)} EN questions to: {OUTPUT_PATH_EN}")

def main():
    print("Starting generation of 50 IAT scenario questions...")
    print(f"Original source path: {ORIGINAL_QUESTIONS_PATH}")
    print(f"ZH output path: {OUTPUT_PATH_ZH}")
    print(f"EN output path: {OUTPUT_PATH_EN}")
    
    call_function = get_gpt5_client()
    seeds = generate_seeds()
    print(f"✓ Generated {len(seeds)} seeds")
    
    original_questions = load_original_questions()
    print(f"✓ Loaded {len(original_questions)} original questions")
    
    existing_questions_zh = load_existing_questions_zh()
    existing_questions_en = load_existing_questions_en()
    existing_ids_zh = {q['id'] for q in existing_questions_zh}
    existing_ids_en = {q['id'] for q in existing_questions_en}
    
    all_questions_zh = existing_questions_zh.copy()
    all_questions_en = existing_questions_en.copy()
    
    lock = threading.Lock()
    
    def process_seed(seed_data):
        seed_id, orig_idx, subject, context, variant = seed_data
        question_id = seed_id + 11
        
        if question_id in existing_ids_zh or question_id in existing_ids_en:
            print(f"Skipping existing question #{question_id}")
            return None
        
        original_q = original_questions[orig_idx]
        adapted_q = generate_adapted_question(seed_id, original_q, subject, context, variant, call_function)
        
        if adapted_q:
            new_q_zh = {
                "id": question_id,
                "scenario": {"subject": subject, "context": context},
                "question": adapted_q['question_zh'],
                "concepts": [{"name": c['name_zh'], "valence": c['valence']} for c in adapted_q['concepts']]
            }
            
            new_q_en = {
                "id": question_id,
                "scenario": {"subject": subject, "context": context},
                "question": adapted_q['question_en'],
                "concepts": [{"name": c['name_en'], "valence": c['valence']} for c in adapted_q['concepts']]
            }
            
            with lock:
                all_questions_zh.append(new_q_zh)
                all_questions_en.append(new_q_en)
            
            print(f"✓ Question #{question_id} generated successfully")
            return question_id
        return None

    print("Beginning multi-threaded generation...")
    max_workers = 5 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {
            executor.submit(process_seed, seed): seed 
            for seed in seeds
            if (seed[0] + 11) not in existing_ids_zh and (seed[0] + 11) not in existing_ids_en
        }
        
        completed_count = 0
        total_tasks = len(future_to_seed)
        
        for future in as_completed(future_to_seed):
            seed_data = future_to_seed[future]
            question_id = seed_data[0] + 11
            try:
                result = future.result()
                if result:
                    completed_count += 1
                    print(f"Progress: {completed_count}/{total_tasks} Question #{result} complete")
                else:
                    print(f"Progress: {completed_count}/{total_tasks} Question #{question_id} failed")
            except Exception as e:
                print(f"Exception for question #{question_id}: {e}")
                completed_count += 1
    
    # Sort and save
    all_questions_zh.sort(key=lambda x: x['id'])
    all_questions_en.sort(key=lambda x: x['id'])
    
    save_questions_zh(all_questions_zh)
    save_questions_en(all_questions_en)
    
    print(f"\n=== FINISHED ===")
    print(f"Total ZH questions: {len(all_questions_zh)}")
    print(f"Total EN questions: {len(all_questions_en)}")

if __name__ == '__main__':
    main()