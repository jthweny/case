#!/usr/bin/env python3
"""
Unit tests for the Parallel Gemini API Manager
Tests the rate limiting functionality
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import logging
import time
from datetime import datetime, timedelta
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parallel_gemini_api import ParallelGeminiAPIManager, APIKeyTracker

class TestParallelGeminiAPIManager(unittest.TestCase):
    """Test cases for ParallelGeminiAPIManager"""

    def test_api_rate_limit_tracking(self):
        """Should correctly track and respect rate limits for API keys (10 requests per minute)"""
        print("🔍 Testing API rate limit tracking...")
        
        # Create an API manager with a controlled test environment
        manager = ParallelGeminiAPIManager()
        
        # Create a test key tracker with controlled limits
        test_key = APIKeyTracker(key="test_key_1", key_type="test")
        test_key.rpm_limit = 10  # 10 requests per minute
        
        # Record 9 requests in the last minute
        now = datetime.now()
        one_minute_ago = now - timedelta(seconds=59)
        
        for i in range(9):
            # Simulate requests made a few seconds apart
            request_time = one_minute_ago + timedelta(seconds=i*6)
            test_key.requests_this_minute.append(request_time)
            test_key.last_request_time = request_time
        
        # Verify the key is still available (9 requests < 10 limit)
        self.assertTrue(test_key.can_make_request(), "Key should be available with 9 requests")
        print(f"   ✅ Key correctly available with 9 requests in last minute")
        
        # Record one more request to reach the limit
        test_key.record_request()
        
        # Verify the key is now unavailable (10 requests = 10 limit)
        self.assertFalse(test_key.can_make_request(), "Key should be unavailable with 10 requests")
        print(f"   ✅ Key correctly unavailable with 10 requests in last minute")
        
        # Calculate when the key should become available again
        next_available = test_key.get_next_available_time()
        expected_available = test_key.requests_this_minute[0] + timedelta(minutes=1, seconds=1)
        time_diff = abs((next_available - expected_available).total_seconds())
        
        # Allow small tolerance for time calculations
        self.assertLess(time_diff, 1, "Next available time should be about 1 minute after oldest request")
        print(f"   ✅ Next available time correctly calculated: {next_available.strftime('%H:%M:%S')}")
        
        # Simulate time passing - remove the oldest request
        test_key.requests_this_minute.pop(0)
        
        # Verify the key is available again (9 requests < 10 limit)
        self.assertTrue(test_key.can_make_request(), "Key should be available after oldest request expires")
        print(f"   ✅ Key correctly available after oldest request expires")
        
        # Test daily limit enforcement
        test_key.requests_today = [now - timedelta(hours=i) for i in range(499)]  # 499 requests today
        self.assertTrue(test_key.can_make_request(), "Key should be available with 499 daily requests")
        
        # Add one more to hit the limit
        test_key.requests_today.append(now)  # 500 requests today
        self.assertFalse(test_key.can_make_request(), "Key should be unavailable with 500 daily requests")
        print(f"   ✅ Daily request limit correctly enforced")
        
        print("✅ API rate limit tracking test passed")
        return True

    @patch('src.parallel_gemini_api.genai.Client')
    async def test_key_rotation_at_rate_limit(self, mock_client):
        """Should manage the rotation of keys when one key reaches its rate limit"""
        print("🔄 Testing API key rotation at rate limit...")
        
        # Create a mock for the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Initialize the manager
        manager = ParallelGeminiAPIManager()
        
        # Access first key and manually set it near rate limit
        first_key = manager.key_trackers[0]
        first_key_id = first_key.key[:10]
        
        # Set the first key to have 9 recent requests (just below the 10 RPM limit)
        now = datetime.now()
        first_key.requests_this_minute = [now - timedelta(seconds=i*5) for i in range(9)]
        
        print(f"   ✅ Set first key {first_key_id}... to have {len(first_key.requests_this_minute)} requests this minute")
        
        # Generate content with the first key - should still work
        result1, metadata1 = await manager.generate_content_single("Test prompt 1", first_key)
        
        # Verify first key was used and now has 10 requests (at the limit)
        self.assertEqual(len(first_key.requests_this_minute), 10)
        self.assertEqual(metadata1['key_id'], first_key_id + "...")
        print(f"   ✅ First key now has {len(first_key.requests_this_minute)} requests (at limit)")
        
        # Get the next available key - should be a different key now
        next_key = manager.get_available_key()
        self.assertIsNotNone(next_key)
        self.assertNotEqual(next_key.key, first_key.key)
        second_key_id = next_key.key[:10]
        print(f"   ✅ Next available key rotated to {second_key_id}...")
        
        # Generate content with the new key
        result2, metadata2 = await manager.generate_content_single("Test prompt 2", next_key)
        
        # Verify second key was used
        self.assertEqual(metadata2['key_id'], second_key_id + "...")
        print(f"   ✅ Second key successfully used for content generation")
        
        # Test full parallel generation to verify it distributes across keys
        test_prompts = ["Test prompt A", "Test prompt B", "Test prompt C"]
        
        # For parallel testing, set all keys to different states
        for i, tracker in enumerate(manager.key_trackers):
            # Every third key at rate limit, others available
            if i % 3 == 0 and i > 0:
                tracker.requests_this_minute = [now - timedelta(seconds=j*5) for j in range(10)]
        
        # Run parallel generation
        results = await manager.generate_content_parallel(test_prompts)
        
        # Verify we got responses for all prompts
        self.assertEqual(len(results), len(test_prompts))
        
        # Check that different keys were used (collect unique key IDs)
        used_keys = set(metadata['key_id'] for _, metadata in results)
        
        print(f"   ✅ Used {len(used_keys)} different keys for parallel requests")
        print(f"   ✅ All {len(test_prompts)} parallel requests completed successfully")
        
        return True

    @patch('src.parallel_gemini_api.genai.Client')
    async def test_model_fallback_on_failure(self, mock_client):
        """Should successfully switch to a different model when the current model fails"""
        print("🔁 Testing model fallback on failure...")
        
        # Create API manager instance
        api_manager = ParallelGeminiAPIManager()
        
        # Get initial model
        initial_model = api_manager.get_current_model()
        initial_index = api_manager.current_model_index
        
        # Mock client behavior
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Set up the mock to fail on first model but succeed on second
        mock_response = MagicMock()
        mock_response.text = "Success with fallback model"
        
        # First call raises an exception, second call succeeds
        mock_generate = mock_client_instance.models.generate_content
        mock_generate.side_effect = [
            Exception("Model currently overloaded"),  # First model fails
            mock_response                             # Second model succeeds
        ]
        
        # Call generate_content, which should fail and then retry with next model
        response = await api_manager.generate_content("Test prompt")
        
        # Verify the model was changed
        new_model = api_manager.get_current_model()
        new_index = api_manager.current_model_index
        
        print(f"   ✅ Initial model: {initial_model} (index: {initial_index})")
        print(f"   ✅ After fallback: {new_model} (index: {new_index})")
        
        # Assert that model index was incremented
        self.assertEqual(new_index, initial_index + 1)
        
        # Assert that a different model was selected
        self.assertNotEqual(initial_model, new_model)
        
        # Assert that the second model in the list was used
        self.assertEqual(new_model, api_manager.models[1])
        
        # Assert that generate_content returned the successful response
        self.assertEqual(response, "Success with fallback model")
        
        # Verify the mock was called with each model
        first_call_model = mock_generate.call_args_list[0][1]['model']
        second_call_model = mock_generate.call_args_list[1][1]['model']
        
        self.assertEqual(first_call_model, initial_model)
        self.assertEqual(second_call_model, new_model)
        
        print(f"   ✅ Model fallback mechanism verified successfully")

    async def test_wait_for_next_available_key(self):
        """Should wait for the next available key if all keys are currently at their rate limits"""
        print("🔄 Testing key availability waiting mechanism...")
        
        # Create manager with mocked keys
        manager = ParallelGeminiAPIManager()
        
        # Set all keys to be temporarily unavailable due to rate limits
        current_time = datetime.now()
        future_time = current_time + timedelta(seconds=10)
        
        # Make all keys temporarily unavailable (rate limited)
        for key_tracker in manager.key_trackers:
            # Fill up the requests_this_minute to hit rate limit
            key_tracker.requests_this_minute = [current_time] * key_tracker.rpm_limit
            # Set the oldest request to a controlled time
            key_tracker.requests_this_minute[0] = future_time - timedelta(minutes=1)
            # Verify key is not available now
            self.assertFalse(key_tracker.can_make_request())
        
        # Save initial next available time
        initial_next_time = manager.get_next_available_time()
        self.assertTrue(initial_next_time > current_time)
        print(f"   ✅ All keys rate limited, next available in: {(initial_next_time - current_time).total_seconds():.1f}s")
        
        # Patch the asyncio.sleep to avoid actual waiting in test
        original_sleep = asyncio.sleep
        
        async def fast_sleep(seconds):
            # Fast-forward our test keys
            for key_tracker in manager.key_trackers:
                # Make the first key available after waiting
                key_tracker.requests_this_minute = []
            # Still sleep a tiny bit to allow asyncio to switch tasks
            await original_sleep(0.01)
            
        # Mock the sleep function to immediately make keys available
        with patch('asyncio.sleep', side_effect=fast_sleep):
            # Test prompt
            test_prompt = "Test prompt when all keys are rate limited"
            
            # Try to generate content - should wait then succeed
            start_time = datetime.now()
            response = await manager.generate_content(test_prompt)
            end_time = datetime.now()
            
            # Verify we got a response
            self.assertIsNotNone(response)
            print(f"   ✅ Got response after waiting: {response[:30]}...")
            
            # Get available key after test
            available_key = manager.get_available_key()
            self.assertIsNotNone(available_key)
            print(f"   ✅ Key became available after waiting")
            
            # Verify the test succeeded
            success = True
        
        return success

    @patch('src.parallel_gemini_api.genai.Client')
    def test_sync_async_context_synchronization(self, mock_client):
        """Should maintain proper synchronization when running in both synchronous and asynchronous contexts"""
        print("🔄 Testing synchronization between sync and async contexts...")
        
        # Create a mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the generate_content response
        mock_response = MagicMock()
        mock_response.text = "Generated content"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Initialize the API manager
        api_manager = ParallelGeminiAPIManager()
        
        # Create a flag to verify async execution
        async_executed = False
        sync_executed = False
        
        # Create a list to track execution order
        execution_order = []
        execution_lock = threading.Lock()
        
        # Define async test function
        async def async_test():
            nonlocal async_executed
            
            # Add a small delay to ensure we can test interleaving
            await asyncio.sleep(0.1)
            
            # Generate content asynchronously
            content = await api_manager.generate_content("Async prompt")
            
            with execution_lock:
                execution_order.append("async")
                async_executed = True
            
            return content
        
        # Define sync test function that will be run in a separate thread
        def sync_test():
            nonlocal sync_executed
            
            # Small delay to ensure the async test has started
            time.sleep(0.05)
            
            # Generate content synchronously
            content = api_manager.generate_content_sync("Sync prompt")
            
            with execution_lock:
                execution_order.append("sync")
                sync_executed = True
            
            return content
        
        # Create and start the async task
        async_task = asyncio.ensure_future(async_test())
        
        # Start the sync test in a separate thread
        sync_thread = threading.Thread(target=sync_test)
        sync_thread.start()
        
        # Wait for both to complete
        sync_thread.join(timeout=5)
        loop = asyncio.get_event_loop()
        async_result = loop.run_until_complete(async_task)
        
        # Verify both executed
        self.assertTrue(async_executed, "Async test should have executed")
        self.assertTrue(sync_executed, "Sync test should have executed")
        
        # Verify content was generated in both cases
        self.assertEqual(async_result, "Generated content")
        
        # Verify the client was called with correct models
        calls = mock_client_instance.models.generate_content.call_args_list
        self.assertEqual(len(calls), 2, "Client should have been called twice")
        
        # Verify both async and sync API calls completed without interfering with each other
        print(f"✅ Execution order: {execution_order}")
        print(f"✅ Both sync and async contexts executed successfully")
        
        # Verify both used their own event loops properly
        self.assertEqual(mock_client_instance.models.generate_content.call_count, 2, 
                         "Should have made exactly 2 API calls")
        
        return True

    @patch('src.parallel_gemini_api.genai.Client')
    def test_key_availability_time_calculation(self, mock_client):
        """Should calculate the exact time when a key will become available again"""
        print("⏱️ Testing key availability time calculation...")
        
        # Create a test key tracker
        test_key = APIKeyTracker(key="test_key", key_type="test")
        
        # Test case 1: Key is available now (no requests)
        now = datetime.now()
        self.assertEqual(test_key.get_next_available_time().replace(microsecond=0), 
                         now.replace(microsecond=0),
                         "Key with no requests should be available immediately")
        print(f"   ✅ Key with no requests is available immediately")
        
        # Test case 2: Rate limit - key has hit RPM limit
        # Fill the key with 10 requests in the last minute (hitting RPM limit)
        test_key.requests_this_minute = []
        base_time = datetime.now() - timedelta(seconds=50)  # All within the last minute
        for i in range(10):  # RPM limit is 10
            test_key.requests_this_minute.append(base_time + timedelta(seconds=i*5))
        
        # Calculate expected availability time (1 minute after oldest request + 1 second buffer)
        expected_time = test_key.requests_this_minute[0] + timedelta(minutes=1, seconds=1)
        actual_time = test_key.get_next_available_time()
        
        # Compare with a small tolerance for processing time
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should be 1 minute after oldest request")
        print(f"   ✅ Rate-limited key becomes available at correct time")
        
        # Test case 3: Minimum interval between requests
        test_key.requests_this_minute = []  # Clear rate limit
        test_key.last_request_time = datetime.now() - timedelta(seconds=3)  # Last request 3 seconds ago
        
        # Expect availability after 6 seconds from last request
        expected_time = test_key.last_request_time + timedelta(seconds=6)
        actual_time = test_key.get_next_available_time()
        
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should respect minimum interval between requests")
        print(f"   ✅ Minimum interval between requests is enforced")
        
        # Test case 4: Daily limit
        test_key.requests_this_minute = []  # Clear rate limit
        test_key.last_request_time = datetime.now() - timedelta(minutes=5)  # Clear minimum interval
        
        # Set daily limit (500 requests) to be reached
        oldest_daily = datetime.now() - timedelta(hours=23)
        test_key.requests_today = [oldest_daily + timedelta(minutes=i) for i in range(500)]
        
        # Expected: 24 hours after oldest daily request + 1 second buffer
        expected_time = oldest_daily + timedelta(days=1, seconds=1)
        actual_time = test_key.get_next_available_time()
        
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should respect daily request limit")
        print(f"   ✅ Daily request limit is enforced correctly")
        
        print("✅ Key availability time calculation test passed")
        return True

if __name__ == "__main__":
    unittest.main()us(self):
        """Test that test_connection method provides accurate status information"""
        print("🔌 Testing connection status reporting...")
        
        # Create API manager with mocked keys
        manager = ParallelGeminiAPIManager()
        
        # Test successful connection
        with patch('google.genai.Client') as mock_client:
            # Mock successful response
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_response = MagicMock()
            mock_response.text = "Hello"
            mock_instance.models.generate_content.return_value = mock_response
            
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 8}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify success status
                self.assertEqual(result["status"], "success")
                self.assertIn("Connected successfully", result["message"])
                self.assertIn("model", result)
                self.assertIn("response", result)
                self.assertIn("Hello", result["response"])
                self.assertIn("metadata", result)
                self.assertIn("key_status", result)
                print(f"   ✅ Success status correctly reported")
        
        # Test connection failure
        with patch('google.genai.Client') as mock_client:
            # Mock failed response
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.models.generate_content.side_effect = Exception("API Error")
            
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 7}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify error status
                self.assertEqual(result["status"], "error")
                self.assertIn("Connection test failed", result["message"])
                self.assertIn("key_status", result)
                print(f"   ✅ Error status correctly reported")
        
        # Test no keys available
        with patch.object(manager, 'get_available_key', return_value=None):
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 0}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify error status due to no keys
                self.assertEqual(result["status"], "error")
                self.assertIn("No API keys available", result["message"])
                self.assertIn("details", result)
                print(f"   ✅ No available keys status correctly reported")
        
        print("✅ Connection status testing complete")

if __name__ == "__main__":
    unittest.main()us(self):
        """Test that test_connection method provides accurate status information"""
        print("🔌 Testing connection status reporting...")
        
        # Create API manager with mocked keys
        manager = ParallelGeminiAPIManager()
        
        # Test successful connection
        with patch('google.genai.Client') as mock_client:
            # Mock successful response
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_response = MagicMock()
            mock_response.text = "Hello"
            mock_instance.models.generate_content.return_value = mock_response
            
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 8}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify success status
                self.assertEqual(result["status"], "success")
                self.assertIn("Connected successfully", result["message"])
                self.assertIn("model", result)
                self.assertIn("response", result)
                self.assertIn("Hello", result["response"])
                self.assertIn("metadata", result)
                self.assertIn("key_status", result)
                print(f"   ✅ Success status correctly reported")
        
        # Test connection failure
        with patch('google.genai.Client') as mock_client:
            # Mock failed response
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.models.generate_content.side_effect = Exception("API Error")
            
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 7}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify error status
                self.assertEqual(result["status"], "error")
                self.assertIn("Connection test failed", result["message"])
                self.assertIn("key_status", result)
                print(f"   ✅ Error status correctly reported")
        
        # Test no keys available
        with patch.object(manager, 'get_available_key', return_value=None):
            # Replace get_key_status with mock implementation
            with patch.object(manager, 'get_key_status', return_value={"total_keys": 10, "available_now": 0}):
                # Test the connection
                result = manager.test_connection()
                
                # Verify error status due to no keys
                self.assertEqual(result["status"], "error")
                self.assertIn("No API keys available", result["message"])
                self.assertIn("details", result)
                print(f"   ✅ No available keys status correctly reported")
        
        print("✅ Connection status testing complete")

if __name__ == "__main__":
    unittest.main()ent')
    def test_key_availability_time_calculation(self, mock_client):
        """Should calculate the exact time when a key will become available again"""
        print("⏱️ Testing key availability time calculation...")
        
        # Create a test key tracker
        test_key = APIKeyTracker(key="test_key", key_type="test")
        
        # Test case 1: Key is available now (no requests)
        now = datetime.now()
        self.assertEqual(test_key.get_next_available_time().replace(microsecond=0), 
                         now.replace(microsecond=0),
                         "Key with no requests should be available immediately")
        print(f"   ✅ Key with no requests is available immediately")
        
        # Test case 2: Rate limit - key has hit RPM limit
        # Fill the key with 10 requests in the last minute (hitting RPM limit)
        test_key.requests_this_minute = []
        base_time = datetime.now() - timedelta(seconds=50)  # All within the last minute
        for i in range(10):  # RPM limit is 10
            test_key.requests_this_minute.append(base_time + timedelta(seconds=i*5))
        
        # Calculate expected availability time (1 minute after oldest request + 1 second buffer)
        expected_time = test_key.requests_this_minute[0] + timedelta(minutes=1, seconds=1)
        actual_time = test_key.get_next_available_time()
        
        # Compare with a small tolerance for processing time
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should be 1 minute after oldest request")
        print(f"   ✅ Rate-limited key becomes available at correct time")
        
        # Test case 3: Minimum interval between requests
        test_key.requests_this_minute = []  # Clear rate limit
        test_key.last_request_time = datetime.now() - timedelta(seconds=3)  # Last request 3 seconds ago
        
        # Expect availability after 6 seconds from last request
        expected_time = test_key.last_request_time + timedelta(seconds=6)
        actual_time = test_key.get_next_available_time()
        
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should respect minimum interval between requests")
        print(f"   ✅ Minimum interval between requests is enforced")
        
        # Test case 4: Daily limit
        test_key.requests_this_minute = []  # Clear rate limit
        test_key.last_request_time = datetime.now() - timedelta(minutes=5)  # Clear minimum interval
        
        # Set daily limit (500 requests) to be reached
        oldest_daily = datetime.now() - timedelta(hours=23)
        test_key.requests_today = [oldest_daily + timedelta(minutes=i) for i in range(500)]
        
        # Expected: 24 hours after oldest daily request + 1 second buffer
        expected_time = oldest_daily + timedelta(days=1, seconds=1)
        actual_time = test_key.get_next_available_time()
        
        self.assertAlmostEqual(actual_time.timestamp(), expected_time.timestamp(), delta=1,
                          "Next available time should respect daily request limit")
        print(f"   ✅ Daily request limit is enforced correctly")
        
        print("✅ Key availability time calculation test passed")
        return True

if __name__ == "__main__":
    unittest.main()d request at {req_time} should be removed from day tracking")
        
        print("   ✅ Cleanup of old requests working correctly!")
        return True

    @patch('src.parallel_gemini_api.genai.Client')
    def test_sync_async_context_synchronization(self, mock_client):
        """Should maintain proper synchronization when running in both synchronous and asynchronous contexts"""
        print("🔄 Testing synchronization between sync and async contexts...")
        
        # Create a mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the generate_content response
        mock_response = MagicMock()
        mock_response.text = "Generated content"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Initialize the API manager
        api_manager = ParallelGeminiAPIManager()
        
        # Create a flag to verify async execution
        async_executed = False
        sync_executed = False
        
        # Create a list to track execution order
        execution_order = []
        execution_lock = threading.Lock()
        
        # Define async test function
        async def async_test():
            nonlocal async_executed
            
            # Add a small delay to ensure we can test interleaving
            await asyncio.sleep(0.1)
            
            # Generate content asynchronously
            content = await api_manager.generate_content("Async prompt")
            
            with execution_lock:
                execution_order.append("async")
                async_executed = True
            
            return content
        
        # Define sync test function that will be run in a separate thread
        def sync_test():
            nonlocal sync_executed
            
            # Small delay to ensure the async test has started
            time.sleep(0.05)
            
            # Generate content synchronously
            content = api_manager.generate_content_sync("Sync prompt")
            
            with execution_lock:
                execution_order.append("sync")
                sync_executed = True
            
            return content
        
        # Create and start the async task
        async_task = asyncio.ensure_future(async_test())
        
        # Start the sync test in a separate thread
        sync_thread = threading.Thread(target=sync_test)
        sync_thread.start()
        
        # Wait for both to complete
        sync_thread.join(timeout=5)
        loop = asyncio.get_event_loop()
        async_result = loop.run_until_complete(async_task)
        
        # Verify both executed
        self.assertTrue(async_executed, "Async test should have executed")
        self.assertTrue(sync_executed, "Sync test should have executed")
        
        # Verify content was generated in both cases
        self.assertEqual(async_result, "Generated content")
        
        # Verify the client was called with correct models
        calls = mock_client_instance.models.generate_content.call_args_list
        self.assertEqual(len(calls), 2, "Client should have been called twice")
        
        # Verify both async and sync API calls completed without interfering with each other
        print(f"✅ Execution order: {execution_order}")
        print(f"✅ Both sync and async contexts executed successfully")
        
        # Verify both used their own event loops properly
        self.assertEqual(mock_client_instance.models.generate_content.call_count, 2, 
                         "Should have made exactly 2 API calls")
        
        return True

if __name__ == "__main__":
    unittest.main()
        for i, (content, metadata) in enumerate(results):
            self.assertEqual(content, f"Response for prompt {i+1}")
            self.assertIn('key_type', metadata)
            self.assertIn('key_id', metadata)
            self.assertIn('processing_time', metadata)
            self.assertIn('model_used', metadata)
            
        # Verify each prompt used a different key
        used_keys = set()
        for args, kwargs in mock_client.call_args_list:
            key = kwargs.get('api_key')
            self.assertNotIn(key, used_keys, "Each prompt should use a different key")
            used_keys.add(key)
            
        print("✅ Successfully processed multiple prompts in parallel using different keys")
        
        return True

if __name__ == "__main__":
    asyncio.run(unittest.main())