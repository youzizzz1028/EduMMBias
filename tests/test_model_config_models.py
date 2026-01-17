#!/usr/bin/env python3
"""
Test all models in the configuration file
Each model is tested with only one data point
"""

import unittest
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_api.client_openai import create_openai_client
from src.model_api import api_utils


class TestAllModels(unittest.TestCase):
    """Test all models in the configuration file"""
    
    @classmethod
    def setUpClass(cls):
        """Load model configuration using api_utils"""
        cls.models_config = api_utils.load_model_config()
        cls.test_prompt = "Please answer in Chinese: What is artificial intelligence? Please give a brief answer."
        
        print(f"Loaded {len(cls.models_config)} models for testing")
    
    def test_all_models(self):
        """Test basic functionality of all models"""
        failed_models = []
        
        for model_key, model_config in self.models_config.items():
            model_name = model_config.get("model_name")
            provider = model_config.get("provider", "Unknown")
            reason_enabled = model_config.get("reason", False)
            
            print(f"\nTesting model: {model_key} ({provider})")
            print(f"Model name: {model_name}")
            print(f"Reasoning mode: {reason_enabled}")
            
            try:
                # Create client
                client, call_func = create_openai_client(model_name=model_name)
                
                # Call model (test with only one data point)
                response = call_func(self.test_prompt)
                
                # Validate response
                self.assertIsNotNone(response, f"Model {model_key} returned empty response")
                self.assertIsInstance(response, str, f"Model {model_key} returned non-string response")
                self.assertGreater(len(response.strip()), 0, f"Model {model_key} returned empty content")
                
                print(f"✓ Model {model_key} test successful")
                print(f"Response length: {len(response)} characters")
                print(f"Response preview: {response[:100]}...")
                
            except Exception as e:
                print(f"✗ Model {model_key} test failed: {e}")
                failed_models.append((model_key, str(e)))
                # Continue testing other models without interrupting the entire test
        
        # Report failed models at the end of the test
        if failed_models:
            failure_msg = f"{len(failed_models)} models failed testing:\n"
            for model_key, error in failed_models:
                failure_msg += f"  - {model_key}: {error}\n"
            self.fail(failure_msg)


def create_individual_model_tests():
    """Create individual test methods for each model"""
    
    # Load model configuration using api_utils
    models_config = api_utils.load_model_config()
    test_prompt = "Please answer in Chinese: What is machine learning? Please give a brief answer."
    
    # Dynamically create test class
    class IndividualModelTests(unittest.TestCase):
        pass
    
    for model_key, model_config in models_config.items():
        model_name = model_config.get("model_name")
        provider = model_config.get("provider", "Unknown")
        
        def create_test_method(model_key, model_name, provider):
            def test_method(self):
                print(f"\nIndividual test for model: {model_key} ({provider})")
                try:
                    client, call_func = create_openai_client(model_name=model_name)
                    response = call_func(test_prompt)
                    
                    self.assertIsNotNone(response)
                    self.assertIsInstance(response, str)
                    self.assertGreater(len(response.strip()), 0)
                    
                    print(f"✓ {model_key} individual test successful")
                    print(f"Response preview: {response[:80]}...")
                    
                except Exception as e:
                    self.fail(f"Model {model_key} test failed: {e}")
            
            return test_method
        
        # Add test method for each model
        test_method = create_test_method(model_key, model_name, provider)
        test_method_name = f"test_{model_key.replace('-', '_')}"
        setattr(IndividualModelTests, test_method_name, test_method)
    
    return IndividualModelTests


# Create class containing individual tests for all models
IndividualModelTests = create_individual_model_tests()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
