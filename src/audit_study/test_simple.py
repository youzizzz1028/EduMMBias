#!/usr/bin/env python3
"""
Simple VLM Testing Script - Used to verify module functionality
"""

import json
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules from the audit study package
from src.audit_study.audit_experiment import AuditStudy, VLMClient


def test_data_loading():
    """Test the data loading functionality"""
    print("Testing data loading...")
    
    study = AuditStudy()
    study.load_configs()
    
    # Check loaded attributes
    print(f"Loaded attributes: {list(study.attribute_pairs.keys())}")
    
    # Check count of student profiles
    print(f"Number of loaded student profiles: {len(study.profile_metadata)}")
    
    # Check count of scenarios
    print(f"Number of loaded scenarios: {len(study.scenarios)}")
    
    # Test sample pair generation logic
    sample_pairs = study.generate_sample_pairs(num_samples_per_pair=1)
    print(f"Number of sample pairs generated: {len(sample_pairs)}")
    
    # Display information for the first sample pair
    if sample_pairs:
        first_pair = sample_pairs[0]
        print(f"First Sample Pair Info:")
        print(f"  Attribute: {first_pair['attribute']}")
        print(f"  Value 1: {first_pair['value1']}")
        print(f"  Value 2: {first_pair['value2']}")
        print(f"  Base Attributes: {first_pair['base_attributes']}")
        print(f"  Student 1 Image Path: {first_pair['profile1']['image_path']}")
        print(f"  Student 2 Image Path: {first_pair['profile2']['image_path']}")
    
    return study, sample_pairs


def test_vlm_client():
    """Test the VLM client initialization (without making actual API calls)"""
    print("\nTesting VLM client initialization...")
    
    try:
        client = VLMClient("gpt-4o")
        print("VLM client initialized successfully.")
        return client
    except Exception as e:
        print(f"VLM client initialization failed: {e}")
        return None


def test_sample_generation():
    """Test the logic for generating sample pairs"""
    print("\nTesting sample generation logic...")
    
    study = AuditStudy()
    study.load_configs()
    
    # Calculate expected number of sample pairs
    combinations = study.get_combination_pairs()
    total_expected_pairs = 0
    
    for attribute, value_pairs in combinations:
        total_expected_pairs += len(value_pairs)
    
    print(f"Number of attribute combinations: {len(combinations)}")
    print(f"Total value pairs across all combinations: {sum(len(vp) for _, vp in combinations)}")
    print(f"Expected sample pairs (1 sample per combo): {total_expected_pairs}")
    
    # Generate actual sample pairs
    sample_pairs = study.generate_sample_pairs(num_samples_per_pair=1)
    print(f"Actual sample pairs generated: {len(sample_pairs)}")
    
    # Verify if image files actually exist on disk
    if sample_pairs:
        first_pair = sample_pairs[0]
        image_path1 = first_pair['profile1']['image_path']
        image_path2 = first_pair['profile2']['image_path']
        
        # Convert to absolute paths for verification
        project_root = Path(__file__).parent.parent.parent
        abs_path1 = project_root / image_path1
        abs_path2 = project_root / image_path2
        
        print(f"Student 1 Absolute Path: {abs_path1}")
        print(f"Student 1 File Exists: {abs_path1.exists()}")
        print(f"Student 2 Absolute Path: {abs_path2}")
        print(f"Student 2 File Exists: {abs_path2.exists()}")


def main():
    """Main testing entry point"""
    print("=" * 60)
    print("VLM Testing Module - Functionality Verification")
    print("=" * 60)
    
    try:
        # 1. Test Data Loading
        study, sample_pairs = test_data_loading()
        
        # 2. Test VLM Client
        client = test_vlm_client()
        
        # 3. Test Sample Generation Logic
        test_sample_generation()
        
        # Calculate total test cases (Pairs * Scenarios)
        total_tests = len(sample_pairs) * len(study.scenarios)
        print(f"\nTotal estimated test cases: {total_tests}")
        
        print("\n" + "=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()