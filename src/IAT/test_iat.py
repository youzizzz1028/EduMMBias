#!/usr/bin/env python3
"""
Test script for IAT experiment
This script tests the basic functionality without making actual API calls
"""

import os
import sys
# Ensure the parent directory is in the path to import iat_experiment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iat_experiment import IATExperiment


def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    # Create experiment instance with dummy model name
    experiment = IATExperiment(model_name="test-model")
    
    # Test attribute pairs loading
    assert experiment.attribute_pairs is not None
    assert 'gender' in experiment.attribute_pairs
    assert 'race' in experiment.attribute_pairs
    print("✓ Attribute pairs loaded successfully")
    
    # Test IAT words loading
    assert experiment.iat_words is not None
    assert 'positive_words' in experiment.iat_words
    assert 'negative_words' in experiment.iat_words
    assert len(experiment.iat_words['positive_words']) == 25
    assert len(experiment.iat_words['negative_words']) == 27  # 27 negative words
    print("✓ IAT words loaded successfully (25 positive, 27 negative)")
    
    # Test profiles loading
    assert len(experiment.profiles) > 0
    print(f"✓ Loaded {len(experiment.profiles)} profiles")
    
    return experiment


def test_attribute_pairs_generation(experiment):
    """Test attribute pairs generation"""
    print("\nTesting attribute pairs generation...")
    
    attribute_pairs = experiment.generate_attribute_pairs()
    assert len(attribute_pairs) > 0
    
    for attr_name, options in attribute_pairs:
        assert len(options) == 2
        assert options[0] != options[1]
        print(f"✓ {attr_name}: {options[0]} vs {options[1]}")
    
    print(f"✓ Generated {len(attribute_pairs)} attribute pairs")
    return attribute_pairs


def test_profile_matching(experiment):
    """Test profile matching functionality"""
    print("\nTesting profile matching...")
    
    # Test with gender attribute
    profile_pairs = experiment.find_matching_profiles('gender', 'male', 'female', n_pairs=1)
    
    if profile_pairs:
        profile1, profile2 = profile_pairs[0]
        
        # Check that profiles differ only in gender
        attrs1 = profile1['attributes']
        attrs2 = profile2['attributes']
        
        # All attributes except gender should be the same
        for attr in ['race', 'socioeconomic_status', 'hobbies', 'health_status']:
            assert attrs1[attr] == attrs2[attr], f"Attribute {attr} differs: {attrs1[attr]} vs {attrs2[attr]}"
        
        # Gender should be different
        assert attrs1['gender'] != attrs2['gender']
        assert {attrs1['gender'], attrs2['gender']} == {'male', 'female'}
        
        print("✓ Profile matching works correctly")
    else:
        print("⚠ No matching profiles found for gender test")


def test_prompt_generation(experiment):
    """Test IAT prompt generation"""
    print("\nTesting prompt generation...")
    
    # Test forward direction
    prompt_forward = experiment.generate_iat_prompt(
        'gender', 'male', 'female', 'love', 'hate', 'forward'
    )
    assert 'love or male' in prompt_forward
    assert 'hate or female' in prompt_forward
    print("✓ Forward prompt generated correctly")
    
    # Test reverse direction
    prompt_reverse = experiment.generate_iat_prompt(
        'gender', 'male', 'female', 'love', 'hate', 'reverse'
    )
    assert 'love or female' in prompt_reverse
    assert 'hate or male' in prompt_reverse
    print("✓ Reverse prompt generated correctly")


def test_response_parsing(experiment):
    """Test model response parsing"""
    print("\nTesting response parsing...")
    
    # Test valid response (using English labels updated in the main script)
    valid_response = """Choice: A
Reason: This image appears to describe a positive concept.
Confidence: 85"""
    
    parsed = experiment.parse_model_response(valid_response)
    assert parsed['choice'] == 'A'
    assert parsed['reason'] == 'This image appears to describe a positive concept.'
    assert parsed['confidence'] == 85
    print("✓ Valid response parsed correctly")
    
    # Test response with variations
    variations = [
        """Choice: B
Reason: Testing reason string.
Confidence: 50""",
        """Choice: A
Reason: Another reason.
Confidence: 100"""
    ]
    
    for response in variations:
        parsed = experiment.parse_model_response(response)
        assert parsed['choice'] in ['A', 'B']
        assert parsed['reason'] is not None
        assert parsed['confidence'] is not None
    print("✓ Response variations parsed correctly")


def test_test_config_generation(experiment):
    """Test test configuration generation"""
    print("\nTesting test configuration generation...")
    
    test_configs = experiment.generate_test_configs()
    assert len(test_configs) > 0
    
    # Check a sample configuration
    if test_configs:
        config = test_configs[0]
        required_fields = [
            'target_attr', 'option1', 'option2', 'positive_word', 
            'negative_word', 'direction', 'image_path', 'image_attributes',
            'profile_id', 'model_name'
        ]
        
        for field in required_fields:
            assert field in config, f"Missing field: {field}"
        
        print(f"✓ Generated {len(test_configs)} test configurations")
        print(f"✓ Sample config: {config['target_attr']} - {config['direction']}")


def main():
    """Run all tests"""
    print("Starting IAT experiment tests...\n")
    
    try:
        # Test configuration loading
        experiment = test_config_loading()
        
        # Test attribute pairs generation
        test_attribute_pairs_generation(experiment)
        
        # Test profile matching
        test_profile_matching(experiment)
        
        # Test prompt generation
        test_prompt_generation(experiment)
        
        # Test response parsing
        test_response_parsing(experiment)
        
        # Test test configuration generation
        test_test_config_generation(experiment)
        
        print("\n🎉 All tests passed!")
        print("\nTo run the actual experiment, use:")
        print("python src/IAT/iat_experiment.py --model <model_name> [--workers <num_workers>]")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()