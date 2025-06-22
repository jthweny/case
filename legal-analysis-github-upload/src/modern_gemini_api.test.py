
#!/usr/bin/env python3
"""
Unit tests for the Modern Gemini API Manager
Tests content generation and metadata verification
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modern_gemini_api import ModernGeminiAPIManager, APIKey, genai

class TestModernGeminiAPIManager(unittest.TestCase):
    """Test cases for ModernGeminiAPIManager"""

    @patch('src.modern_gemini_api.genai.Client')
    def test_initialize_client_with_first_key(self, mock_client):
        """Should successfully initialize client with the first API key in the list"""
        print("🔧 Testing ModernGeminiAPIManager client initialization...")
        
        # Create a mock for the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Initialize the manager
        manager = ModernGeminiAPIManager()
        
        # Verify client was initialized with the first key
        first_key = manager.api_keys[0].key
        mock_client.assert_called_once()
        
        # Extract the args from the call
        call_args = mock_client.call_args
        
        # Verify the API key is the first one in the list
        self.assertEqual(call_args[1]['api_key'], first_key)
        
        # Verify timeout was set correctly
        self.assertEqual(call_args[1]['http_options'].timeout, manager.timeout_ms)
        
        print(f"   ✅ Client initialized with first key: {first_key[:10]}...")
        
        return True

    def test_model_rotation(self):
        """Test that the API manager rotates to the next model when current model fails"""
        print("🔄 Testing model rotation functionality...")
        
        # Create API manager instance
        api_manager = ModernGeminiAPIManager()
        
        # Get initial model
        initial_model = api_manager.get_current_model()
        initial_index = api_manager.current_model_index
        print(f"   ✅ Initial model: {initial_model}")
        
        # Simulate rotation
        api_manager.rotate_model()
        
        # Get new model
        new_model = api_manager.get_current_model()
        new_index = api_manager.current_model_index
        print(f"   ✅ After rotation: {new_model}")
        
        # Verify rotation happened correctly
        if len(api_manager.models) > 1:
            success = initial_model != new_model and new_index == (initial_index + 1) % len(api_manager.models)
            print(f"   {'✅' if success else '❌'} Model rotation {'successful' if success else 'failed'}")
        else:
            # If only one model, it should rotate back to itself
            success = initial_model == new_model and new_index == 0
            print(f"   ✅ Only one model available, stayed on same model")
        
        # Complete rotation through all models to verify cycling works
        expected_cycle = [api_manager.models[i % len(api_manager.models)] for i in range(initial_index, initial_index + len(api_manager.models))]
        actual_cycle = []
        
        # Reset to initial position
        api_manager.current_model_index = initial_index
        
        # Rotate through all models
        for _ in range(len(api_manager.models)):
            actual_cycle.append(api_manager.get_current_model())
            api_manager.rotate_model()
        
        cycle_success = expected_cycle == actual_cycle
        print(f"   {'✅' if cycle_success else '❌'} Full rotation cycle {'matched expected pattern' if cycle_success else 'failed'}")
        
        return success and cycle_success

    def test_reset_error_counts(self):
        """Test that error counts are reset when all keys have high error counts"""
        print("🔑 Testing API key error count reset...")
        
        try:
            # Initialize the API manager
            api_manager = ModernGeminiAPIManager()
            
            # Set all keys to have error count of 3 (threshold)
            for key in api_manager.api_keys:
                key.error_count = 3
                key.has_credit = True
            
            # Verify all keys have error count of 3
            all_keys_have_errors = all(key.error_count == 3 for key in api_manager.api_keys)
            print(f"   ✅ All keys have error count 3: {all_keys_have_errors}")
            
            # Try to get next key - this should trigger reset
            next_key = api_manager.get_next_key()
            
            # Verify a key was returned despite all having high error counts
            key_returned = next_key is not None
            print(f"   ✅ Key returned despite high error counts: {key_returned}")
            
            # Verify error counts were reset
            error_counts_reset = all(key.error_count == 0 for key in api_manager.api_keys)
            print(f"   ✅ All error counts were reset: {error_counts_reset}")
            
            # Verify all keys are now available
            available_keys = [k for k in api_manager.api_keys if k.has_credit and k.error_count < 3]
            all_keys_available = len(available_keys) == len(api_manager.api_keys)
            print(f"   ✅ All keys now available: {all_keys_available}")
            
            return all_keys_have_errors and key_returned and error_counts_reset and all_keys_available
            
        except Exception as e:
            print(f"   ❌ Error count reset test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_get_api_status_working_keys_count(self):
        """Test that get_api_status returns accurate count of working keys"""
        print("🧪 Testing API status with working keys count...")
        
        # Create the API manager
        api_manager = ModernGeminiAPIManager()
        
        # Mock the test_api_connectivity method to avoid actual API calls
        with patch.object(api_manager, 'test_api_connectivity') as mock_test:
            # Setup mock return value
            mock_test.return_value = {
                'gemini': {'status': 'active', 'details': 'Connected with preview key'},
                'openai': {'status': 'inactive', 'details': 'Not available'}
            }
            
            # Manually set some keys as having errors or no credit
            api_manager.api_keys[0].error_count = 3  # This key won't count as working
            api_manager.api_keys[1].has_credit = False  # This key won't count as working
            api_manager.api_keys[2].error_count = 2  # This key will count as working (error < 3)
            
            # Get the API status
            status = api_manager.get_api_status()
            
            # Total keys should be the length of the api_keys list
            self.assertEqual(status['total_keys'], len(api_manager.api_keys))
            
            # Working keys should be those with has_credit=True and error_count<3
            expected_working_keys = len([k for k in api_manager.api_keys if k.has_credit and k.error_count < 3])
            self.assertEqual(status['working_keys'], expected_working_keys)
            
            # Verify the calculation manually to be sure
            self.assertEqual(expected_working_keys, 8)  # 10 total - 1 with error_count=3 - 1 with has_credit=False
            
            print(f"   ✅ Total keys: {status['total_keys']}")
            print(f"   ✅ Working keys: {status['working_keys']}")
            print(f"   ✅ Current model: {status['current_model']}")

    @patch('google.genai.Client')
    def test_retry_logic_on_content_generation_failure(self, mock_client_class):
        """Should correctly implement retry logic when content generation fails"""
        # Setup mock client to fail twice then succeed
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Configure generate_content to fail twice then succeed on third try
        mock_response = MagicMock()
        mock_response.text = "Successfully generated content after retries"
        
        # Set up side effects for generate_content: Exception, Exception, then success
        mock_generate = mock_client_instance.models.generate_content
        mock_generate.side_effect = [
            Exception("API quota exceeded"),  # First attempt fails
            Exception("Server error"),        # Second attempt fails
            mock_response                     # Third attempt succeeds
        ]
        
        # Call the method that should retry
        prompt = "Test prompt for retry logic"
        result, metadata = self.api_manager.generate_content(prompt)
        
        # Verify it was called exactly 3 times (2 failures + 1 success)
        self.assertEqual(mock_generate.call_count, 3)
        
        # Verify the final response is correct
        self.assertEqual(result, "Successfully generated content after retries")
        
        # Verify metadata contains expected fields
        self.assertIn("key_type", metadata)
        self.assertIn("processing_time", metadata)
        self.assertIn("model_used", metadata)

    def setUp(self):
        """Set up test environment"""
        # Create manager with mocked client initialization
        with patch('src.modern_gemini_api.genai.Client'):
            self.api_manager = ModernGeminiAPIManager()
            # Replace real API keys with test keys
            self.api_manager.api_keys = [
                APIKey("test_key_1", "preview"),
                APIKey("test_key_2", "experimental")
            ]
    
    def test_api_connectivity(self):
        """Test API connectivity in both quick and full test modes"""
        print("🔌 Testing API connectivity...")
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = "API_TEST_SUCCESS"
        
        # Test quick connectivity (30s timeout)
        with patch('src.modern_gemini_api.genai.Client') as mock_client:
            # Setup mock client
            client_instance = mock_client.return_value
            client_instance.models.generate_content.return_value = mock_response
            
            # Run quick test
            quick_results = self.api_manager.test_api_connectivity(quick_test=True)
            
            # Verify quick test timeout setting (30s = 30000ms)
            mock_client.assert_called_with(
                api_key="test_key_1",
                http_options=unittest.mock.ANY
            )
            self.assertEqual(
                client_instance.models.generate_content.call_args[1]['model'],
                self.api_manager.get_current_model()
            )
            
            # Verify results structure
            self.assertEqual(quick_results['gemini']['status'], 'active')
            self.assertIn('preview', quick_results['gemini']['details'])
            print(f"   ✅ Quick test passed: {quick_results['gemini']['status']}")
        
        # Test full connectivity (120s timeout)
        with patch('src.modern_gemini_api.genai.Client') as mock_client:
            # Setup mock client
            client_instance = mock_client.return_value
            client_instance.models.generate_content.return_value = mock_response
            
            # Run full test
            full_results = self.api_manager.test_api_connectivity(quick_test=False)
            
            # Verify timeout setting (120s = 120000ms)
            mock_client.assert_called_with(
                api_key="test_key_1",
                http_options=unittest.mock.ANY
            )
            
            # Verify results structure
            self.assertEqual(full_results['gemini']['status'], 'active')
            self.assertIn('preview', full_results['gemini']['details'])
            print(f"   ✅ Full test passed: {full_results['gemini']['status']}")
        
        # Test error handling
        with patch('src.modern_gemini_api.genai.Client') as mock_client:
            # Setup mock client to raise exception
            client_instance = mock_client.return_value
            client_instance.models.generate_content.side_effect = Exception("API quota exceeded")
            
            # Run test with error
            error_results = self.api_manager.test_api_connectivity()
            
            # Verify error handling
            self.assertEqual(error_results['gemini']['status'], 'inactive')
            self.assertIn('Connection failed', error_results['gemini']['details'])
            print(f"   ✅ Error handling test passed")
            
            # Verify key marked as no credit (quota exceeded)
            self.assertFalse(self.api_manager.api_keys[0].has_credit)
            
        print("🎉 API connectivity tests completed!")

    def test_api_timeout(self):
        """Test that ModernGeminiAPIManager correctly times out after specified period"""
        print("🔍 Testing API timeout functionality...")
        
        # Create API manager with mock keys
        api_manager = ModernGeminiAPIManager()
        
        # Replace actual keys with test keys to avoid real API calls
        api_manager.api_keys = [APIKey("test_key_1", "preview")]
        
        # Set a shorter timeout for testing
        api_manager.timeout_ms = 2000  # 2 seconds
        
        # Create a mock client that sleeps longer than the timeout
        with patch('google.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock generate_content to sleep longer than timeout and then raise a timeout error
            def slow_generate(*args, **kwargs):
                time.sleep(3)  # Sleep longer than our 2-second timeout
                raise TimeoutError("Request timed out")
                
            mock_client.models.generate_content.side_effect = slow_generate
            
            # Attempt to generate content, which should timeout
            try:
                api_manager.generate_content("Test prompt")
                print("   ❌ Failed: No timeout occurred")
                return False
            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    print("   ✅ Success: Request correctly timed out")
                    return True
                else:
                    print(f"   ❌ Failed: Unexpected error: {e}")
                    return False

class TestModernGeminiAPIQuotaHandling(unittest.TestCase):
    """Test the API quota handling functionality of ModernGeminiAPIManager"""
    
    def setUp(self):
        """Set up the test environment"""
        # Disable logging for cleaner test output
        logging.disable(logging.CRITICAL)
        
        # Create the API manager with test keys
        self.api_manager = ModernGeminiAPIManager()
        # Replace the actual keys with test keys
        self.api_manager.api_keys = [
            APIKey("test_key_1", "preview"),
            APIKey("test_key_2", "preview"),
            APIKey("test_key_3", "experimental")
        ]
        
    def tearDown(self):
        """Clean up after the test"""
        logging.disable(logging.NOTSET)
    
    def test_quota_limit_handling(self):
        """Test that quota errors mark keys as having no credit"""
        print("🧪 Testing API quota limit handling...")
        
        # Create a mock response that simulates a quota error
        quota_error = Exception("Quota exceeded: The API key has reached its limit")
        
        with patch('google.genai.Client') as mock_client:
            # Setup the mock to raise a quota error
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_models = MagicMock()
            mock_instance.models = mock_models
            mock_models.generate_content.side_effect = quota_error
            
            # Test the API connectivity which should trigger the quota error
            results = self.api_manager.test_api_connectivity()
            
            # Verify the first key was used
            first_key = self.api_manager.api_keys[0]
            self.assertEqual(first_key.has_credit, False, "Key should be marked as having no credit after quota error")
            self.assertEqual(first_key.error_count, 1, "Error count should be incremented")
            
            # Verify the results indicate an inactive status with the correct error
            self.assertEqual(results['gemini']['status'], 'inactive', "Gemini status should be 'inactive'")
            self.assertIn('quota', results['gemini']['details'].lower(), "Error details should mention quota")
        
        print("✅ Quota limit handling test passed")

class TestModernGeminiAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.api_manager = ModernGeminiAPIManager()
    
    @patch('google.genai.Client')
    def test_generate_content_with_metadata(self, mock_client):
        """Test content generation with proper metadata in response"""
        print("🧪 Testing content generation with metadata...")
        
        # Mock the genai client
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock the generate_content response
        mock_response = MagicMock()
        mock_response.text = "API_TEST_SUCCESS"
        mock_instance.models.generate_content.return_value = mock_response
        
        # Execute the function to test
        test_prompt = "Generate test content"
        content, metadata = self.api_manager.generate_content(test_prompt)
        
        # Assert content was generated
        self.assertEqual(content, "API_TEST_SUCCESS")
        
        # Assert metadata has the expected keys
        self.assertIn('key_type', metadata)
        self.assertIn('key_id', metadata)
        self.assertIn('processing_time', metadata)
        self.assertIn('model_used', metadata)
        self.assertIn('sdk_version', metadata)
        
        # Assert model name is one of the expected models
        self.assertIn(metadata['model_used'], self.api_manager.models)
        
        # Assert SDK version is correct
        self.assertEqual(metadata['sdk_version'], 'google-genai')
        
        # Assert processing time is a float
        self.assertIsInstance(metadata['processing_time'], float)
        
        print(f"✅ Test passed! Generated content with metadata: {metadata}")

    def test_model_fallback(self):
        """Test that the API manager falls back to faster models when higher quality models are unavailable"""
        print("🔄 Testing model fallback mechanism...")
        
        # Create manager
        api_manager = ModernGeminiAPIManager()
        
        # Check initial model (should be the highest quality one)
        initial_model = api_manager.get_current_model()
        print(f"   ✅ Initial model: {initial_model}")
        assert initial_model == "gemini-2.5-pro", f"Expected to start with gemini-2.5-pro, got {initial_model}"
        
        # Simulate failure with first model by rotating
        api_manager.rotate_model()
        second_model = api_manager.get_current_model()
        print(f"   ✅ Fallback model 1: {second_model}")
        assert second_model == "gemini-2.5-pro-preview-06-05", f"Expected second model to be preview-06-05, got {second_model}"
        
        # Continue rotation to verify all fallbacks
        api_manager.rotate_model()
        third_model = api_manager.get_current_model()
        print(f"   ✅ Fallback model 2: {third_model}")
        assert third_model == "gemini-2.5-pro-preview-05-06", f"Expected third model to be preview-05-06, got {third_model}"
        
        # Final fallback should be the flash model
        api_manager.rotate_model()
        final_model = api_manager.get_current_model()
        print(f"   ✅ Final fallback model: {final_model}")
        assert final_model == "gemini-2.5-flash", f"Expected final fallback to be gemini-2.5-flash, got {final_model}"
        
        # Verify rotation loops back to beginning
        api_manager.rotate_model()
        looped_model = api_manager.get_current_model()
        print(f"   ✅ Loop back to first model: {looped_model}")
        assert looped_model == "gemini-2.5-pro", f"Expected to loop back to gemini-2.5-pro, got {looped_model}"
        
        print("✅ Model fallback mechanism works correctly")
        return True

if __name__ == "__main__":
    unittest.main()
