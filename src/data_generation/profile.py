#!/usr/bin/env python3
"""
Profile generation module for creating profile images based on attribute combinations.
Reads attribute pairs from configs/attribute_pairs.yaml and generates profile images
with corresponding prompts and metadata.
"""

import yaml
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import uuid
import concurrent.futures
from tqdm import tqdm

# Import image generation functionality
try:
    from ..model_api.image_generation import create_image_generation_client
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.model_api.image_generation import create_image_generation_client


def load_attribute_pairs() -> Dict[str, List[str]]:
    """
    Load attribute pairs from config file.
    
    Returns:
        Dict[str, List[str]]: Dictionary of attribute names and their possible values
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "attribute_pairs.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Attribute pairs config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        attribute_pairs = yaml.safe_load(f)
    
    return attribute_pairs


def generate_attribute_combinations(attribute_pairs: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Generate all possible combinations of attribute values.
    
    Args:
        attribute_pairs: Dictionary of attribute names and their possible values
        
    Returns:
        List[Dict[str, str]]: List of dictionaries representing each combination
    """
    # Get attribute names and their possible values
    attribute_names = list(attribute_pairs.keys())
    attribute_values = [attribute_pairs[name] for name in attribute_names]
    
    # Generate all combinations
    combinations = list(itertools.product(*attribute_values))
    
    # Convert to list of dictionaries
    result = []
    for combo in combinations:
        combination_dict = {}
        for i, attr_name in enumerate(attribute_names):
            combination_dict[attr_name] = combo[i]
        result.append(combination_dict)
    
    return result


def create_fluent_description(attributes: Dict[str, str]) -> str:
    """
    Create a fluent English description from attribute combinations.
    
    Args:
        attributes: Dictionary of attribute values
        
    Returns:
        str: Fluent English description for image generation prompt
    """
    # Base description template
    description_parts = []
    
    # Add gender and race
    if 'gender' in attributes and 'race' in attributes:
        description_parts.append(f"A {attributes['gender']} {attributes['race']} person")
    
    # Add socioeconomic status
    if 'socioeconomic_status' in attributes:
        status = attributes['socioeconomic_status']
        if status == 'low':
            description_parts.append("from a low-income background")
        elif status == 'middle':
            description_parts.append("from a middle-class background")
        elif status == 'high':
            description_parts.append("from a high-income background")
    
    # Add hobbies
    if 'hobbies' in attributes:
        hobbies = attributes['hobbies']
        if hobbies == 'sports and travel':
            description_parts.append("who enjoys sports and traveling")
        elif hobbies == 'arts':
            description_parts.append("who is passionate about arts")
        elif hobbies == 'technology':
            description_parts.append("who is interested in technology")
        elif hobbies == 'reading':
            description_parts.append("who loves reading")
        elif hobbies == 'gaming':
            description_parts.append("who enjoys gaming")
    
    # Add health status
    if 'health_status' in attributes:
        health = attributes['health_status']
        if health == 'excellent':
            description_parts.append("with excellent health")
        elif health == 'fair':
            description_parts.append("with fair health")
        elif health == 'poor':
            description_parts.append("with poor health")
    
    # Combine all parts
    description = " ".join(description_parts)
    
    # Add photo style description
    description += ". student portrait photo, realistic, high quality, detailed facial features."
    
    return description


def generate_unique_filename(attributes: Dict[str, str], iteration: int = None) -> str:
    """
    Generate a unique filename based on attribute combination.
    
    Args:
        attributes: Dictionary of attribute values
        iteration: Optional iteration number for multiple images per profile
        
    Returns:
        str: Unique filename for the image
    """
    # Create a string representation of the attributes
    attr_string = "_".join([f"{k}_{v.replace(' ', '_')}" for k, v in sorted(attributes.items())])
    
    # Add timestamp and UUID for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # Include iteration number if provided
    if iteration is not None:
        return f"profile_{attr_string}_{timestamp}_{unique_id}_iter{iteration:02d}.png"
    else:
        return f"profile_{attr_string}_{timestamp}_{unique_id}.png"


def save_profile_metadata(attributes: Dict[str, str], prompt: str, image_path: str, 
                         metadata_file: Path, success: bool = True) -> None:
    """
    Save profile metadata to a JSONL file (JSON Lines format).
    Uses append mode to avoid file corruption in concurrent writes.
    
    Args:
        attributes: Dictionary of attribute values
        prompt: Generated prompt used for image creation
        image_path: Path to the generated image (absolute path)
        metadata_file: Path to the metadata JSONL file
        success: Whether the image generation was successful
    """
    # Convert absolute path to relative path from project root
    project_root = Path(__file__).parent.parent.parent
    relative_image_path = Path(image_path).relative_to(project_root)
    
    metadata = {
        "attributes": attributes,
        "prompt": prompt,
        "image_path": str(relative_image_path),
        "generated_at": datetime.now().isoformat(),
        "unique_id": str(uuid.uuid4()),
        "success": success
    }
    
    # Ensure directory exists
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to JSONL file (one JSON object per line)
    with open(metadata_file, 'a', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
        f.write('\n')


def _generate_single_profile(attributes: Dict[str, str], output_dir: Path, metadata_file: Path, 
                           generate_image_func, images_per_profile: int = 3) -> Dict:
    """
    Generate multiple profile images for a single attribute combination.
    
    Args:
        attributes: Attribute combination to generate
        output_dir: Directory to save the images
        metadata_file: Path to save metadata
        generate_image_func: Image generation function
        images_per_profile: Number of images to generate per profile (default: 3)
        
    Returns:
        Dict: Generation result information
    """
    try:
        # Generate fluent description
        prompt = create_fluent_description(attributes)
        
        generated_paths = []
        
        # Generate multiple images
        for i in range(images_per_profile):
            # Generate unique filename with iteration index
            filename = generate_unique_filename(attributes, iteration=i)
            image_path = output_dir / filename
            
            # Generate image with natural style for realistic student photos
            generated_path = generate_image_func(
                prompt=prompt,
                save_path=str(image_path),
                size="1024x1024",
                quality="standard",
                style="natural"
            )
            generated_paths.append(generated_path)
            
            # Save metadata for each image
            save_profile_metadata(attributes, prompt, generated_path, metadata_file, success=True)
        
        return {
            "attributes": attributes,
            "prompt": prompt,
            "image_paths": [str(path) for path in generated_paths],
            "images_generated": len(generated_paths),
            "success": True
        }
        
    except Exception as e:
        return {
            "attributes": attributes,
            "error": str(e),
            "success": False
        }


def get_completed_combinations(metadata_file: Path, images_per_profile: int = 3) -> List[Dict]:
    """
    Get list of attribute combinations that have already been successfully generated.
    
    Args:
        metadata_file: Path to the metadata JSONL file
        images_per_profile: Number of images expected per profile (default: 3)
        
    Returns:
        List[Dict]: List of successfully generated attribute combinations
    """
    if not metadata_file.exists():
        return []
    
    try:
        # Read JSONL file (one JSON object per line)
        existing_data = []
        total_lines = 0
        valid_lines = 0
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    existing_data.append(item)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        # Count occurrences of each attribute combination
        combination_counts = {}
        
        # Count entries for debugging
        total_entries = len(existing_data)
        valid_entries = 0
        
        for item in existing_data:
            # Skip if not a dictionary
            if not isinstance(item, dict):
                continue
                
            # Skip if missing attributes
            if 'attributes' not in item:
                continue
                
            valid_entries += 1
            
            # Check if the item is successful (has success=True or no success field)
            # If success field is missing, assume it was successful (backward compatibility)
            is_successful = item.get('success', True)
            
            if is_successful:
                # Create a hashable representation of the attributes
                try:
                    attr_key = tuple(sorted(item['attributes'].items()))
                    combination_counts[attr_key] = combination_counts.get(attr_key, 0) + 1
                except (TypeError, AttributeError) as e:
                    print(f"Warning: Error processing attributes {item.get('attributes')}: {e}")
                    continue
        
        # Only consider combinations that have the expected number of images
        completed_combinations = []
        for attr_key, count in combination_counts.items():
            if count >= images_per_profile:
                # Convert back to dictionary
                completed_combinations.append(dict(attr_key))
        
        print(f"Debug: Read {valid_lines}/{total_lines} valid lines, processed {valid_entries}/{total_entries} valid entries")
        print(f"Debug: Found {len(combination_counts)} unique combinations in metadata")
        print(f"Debug: {len(completed_combinations)} combinations have {images_per_profile}+ images")
        
        return completed_combinations
    
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Error reading metadata file, starting fresh: {e}")
        return []


def generate_profile_images(output_dir: Path = None, metadata_file: Path = None, 
                          max_workers: int = 5, images_per_profile: int = 3, 
                          resume: bool = True) -> List[Dict]:
    """
    Generate profile images for all attribute combinations using multithreading.
    
    Args:
        output_dir: Directory to save generated images (default: data/generated/profile_images)
        metadata_file: Path to save metadata (default: data/generated/profile_metadata.json)
        max_workers: Maximum number of worker threads (default: 5)
        images_per_profile: Number of images to generate per profile (default: 3)
        resume: Whether to resume from previous run (default: True)
        
    Returns:
        List[Dict]: List of generated profile information
    """
    # Set default paths
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "generated" / "profile_images"
    
    if metadata_file is None:
        metadata_file = Path(__file__).parent.parent.parent / "data" / "generated" / "profile_metadata.jsonl"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load attribute pairs
    attribute_pairs = load_attribute_pairs()
    print(f"Loaded {len(attribute_pairs)} attributes")
    
    # Generate all combinations
    all_combinations = generate_attribute_combinations(attribute_pairs)
    print(f"Generated {len(all_combinations)} attribute combinations")
    
    # Get completed combinations if resuming
    if resume and metadata_file.exists():
        completed_combinations = get_completed_combinations(metadata_file, images_per_profile)
        print(f"Found {len(completed_combinations)} previously completed combinations")
        
        # Filter out completed combinations
        remaining_combinations = []
        completed_set = {tuple(sorted(c.items())) for c in completed_combinations}
        
        for combo in all_combinations:
            combo_key = tuple(sorted(combo.items()))
            if combo_key not in completed_set:
                remaining_combinations.append(combo)
        
        print(f"Remaining combinations to process: {len(remaining_combinations)}")
        combinations = remaining_combinations
    else:
        combinations = all_combinations
        print(f"Starting fresh with {len(combinations)} combinations")
    
    if len(combinations) == 0:
        print("All combinations already completed. Nothing to generate.")
        return []
    
    print(f"Generating {images_per_profile} images per profile")
    print(f"Total images to generate: {len(combinations) * images_per_profile}")
    
    # Create image generation client
    generate_image = create_image_generation_client()
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_attributes = {
            executor.submit(_generate_single_profile, attributes, output_dir, metadata_file, generate_image, images_per_profile): attributes
            for attributes in combinations
        }
        
        # Process results with progress bar
        results = []
        with tqdm(total=len(combinations), desc="Generating profile images") as pbar:
            for future in concurrent.futures.as_completed(future_to_attributes):
                attributes = future_to_attributes[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        pbar.set_postfix_str(f"✓ {attributes} ({result['images_generated']} images)")
                    else:
                        pbar.set_postfix_str(f"✗ {attributes}: {result['error']}")
                except Exception as e:
                    result = {
                        "attributes": attributes,
                        "error": str(e),
                        "success": False
                    }
                    results.append(result)
                    pbar.set_postfix_str(f"✗ {attributes}: {e}")
                finally:
                    pbar.update(1)
    
    # Print summary
    successful = len([r for r in results if r['success']])
    failed = len([r for r in results if not r['success']])
    total_images_generated = sum([r.get('images_generated', 0) for r in results if r['success']])
    
    print(f"\nProfile generation completed.")
    print(f"Successful profiles: {successful}")
    print(f"Failed profiles: {failed}")
    print(f"Total images generated in this session: {total_images_generated}")
    
    # Get total completed count including previous runs
    if resume and metadata_file.exists():
        total_completed = len(get_completed_combinations(metadata_file))
        print(f"Total completed profiles (including previous runs): {total_completed}")
    
    if failed > 0:
        print("\nFailed combinations:")
        for result in results:
            if not result['success']:
                print(f"  {result['attributes']}: {result['error']}")
    
    return results


def generate_specific_profile(attributes: Dict[str, str], output_dir: Path = None, 
                            images_per_profile: int = 3) -> Dict:
    """
    Generate profile images for a specific attribute combination.
    
    Args:
        attributes: Specific attribute combination to generate
        output_dir: Directory to save the images
        images_per_profile: Number of images to generate (default: 3)
        
    Returns:
        Dict: Generation result information
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "generated" / "profile_images"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate attributes against available pairs
    attribute_pairs = load_attribute_pairs()
    for attr_name, attr_value in attributes.items():
        if attr_name not in attribute_pairs:
            raise ValueError(f"Unknown attribute: {attr_name}")
        if attr_value not in attribute_pairs[attr_name]:
            raise ValueError(f"Invalid value '{attr_value}' for attribute '{attr_name}'")
    
    # Generate image
    generate_image = create_image_generation_client()
    prompt = create_fluent_description(attributes)
    
    generated_paths = []
    
    try:
        # Generate multiple images
        for i in range(images_per_profile):
            filename = generate_unique_filename(attributes, iteration=i)
            image_path = output_dir / filename
            
            generated_path = generate_image(
                prompt=prompt,
                save_path=str(image_path),
                size="1024x1024",
                quality="standard",
                style="natural"
            )
            generated_paths.append(generated_path)
            
            # Save metadata for each image
            metadata_file = Path(__file__).parent.parent.parent / "data" / "generated" / "profile_metadata.jsonl"
            save_profile_metadata(attributes, prompt, generated_path, metadata_file)
        
        return {
            "attributes": attributes,
            "prompt": prompt,
            "image_paths": [str(path) for path in generated_paths],
            "images_generated": len(generated_paths),
            "success": True
        }
        
    except Exception as e:
        return {
            "attributes": attributes,
            "error": str(e),
            "success": False
        }


if __name__ == "__main__":
    # Test the profile generation
    try:
        print("Starting profile image generation...")
        
        # Generate all combinations
        results = generate_profile_images()
        
        # Print summary
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])
        
        print(f"\nGeneration Summary:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed combinations:")
            for result in results:
                if not result['success']:
                    print(f"  {result['attributes']}: {result['error']}")
        
    except Exception as e:
        print(f"Error during profile generation: {e}")
