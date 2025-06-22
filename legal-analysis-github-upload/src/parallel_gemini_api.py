#!/usr/bin/env python3
"""
Advanced Parallel Gemini API Manager with Precise Rate Limit Tracking
Implements concurrent processing with exact timing for maximum efficiency
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
from google import genai
from google.genai.types import HttpOptions
import streamlit as st

@dataclass
class APIKeyTracker:
    """Tracks precise usage timing for an API key"""
    key: str
    key_type: str
    
    # Rate limits for Gemini 2.5 Flash Free Tier (OFFICIAL from Google)
    rpm_limit: int = 10  # 10 requests per minute (OFFICIAL)
    tpm_limit: int = 250000  # 250,000 tokens per minute (OFFICIAL)
    rpd_limit: int = 500  # 500 requests per day (OFFICIAL)
    
    # Tracking data
    requests_this_minute: List[datetime] = field(default_factory=list)
    requests_today: List[datetime] = field(default_factory=list)
    tokens_this_minute: int = 0
    last_request_time: Optional[datetime] = None
    is_available: bool = True
    error_count: int = 0
    
    def get_next_available_time(self) -> datetime:
        """Calculate exact time when this key will be available"""
        now = datetime.now()
        
        # Clean old requests (older than 1 minute)
        cutoff_minute = now - timedelta(minutes=1)
        self.requests_this_minute = [req for req in self.requests_this_minute if req > cutoff_minute]
        
        # Clean old daily requests
        cutoff_day = now - timedelta(days=1)
        self.requests_today = [req for req in self.requests_today if req > cutoff_day]
        
        # Check if we've hit RPM limit
        if len(self.requests_this_minute) >= self.rpm_limit:
            # Find when we can make the next request
            oldest_request = min(self.requests_this_minute)
            return oldest_request + timedelta(minutes=1, seconds=1)  # 1 second buffer
        
        # Check if we've hit daily limit
        if len(self.requests_today) >= self.rpd_limit:
            # Find when daily limit resets
            oldest_daily = min(self.requests_today)
            return oldest_daily + timedelta(days=1, seconds=1)  # 1 second buffer
        
        # If no limits hit, available now
        return now
    
    def can_make_request(self) -> bool:
        """Check if this key can make a request right now"""
        return self.get_next_available_time() <= datetime.now()
    
    def record_request(self, tokens_used: int = 0):
        """Record a successful request"""
        now = datetime.now()
        self.requests_this_minute.append(now)
        self.requests_today.append(now)
        self.tokens_this_minute += tokens_used
        self.last_request_time = now
        self.error_count = 0  # Reset error count on success
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
        if self.error_count >= 3:
            self.is_available = False

class ParallelGeminiAPIManager:
    """
    Advanced parallel API manager with precise rate limit tracking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize key trackers
        api_keys = [
            ("AIzaSyBn_mF1-j6rckpYdEqtKc4VzvO53PkRJHk", "preview"),
            ("AIzaSyDvSLlG56zNrCGWKv-lL3pCYlp5CO47jog", "preview"),
            ("AIzaSyB1bc8KJ_BBbsEoKyndiABIFwKvV6KhM6Y", "preview"),
            ("AIzaSyAamPAkIms_pl87n5jsAHojtixLJ1teGaU", "preview"),
            ("AIzaSyCPz_EiSt4-1jFWvRrSS95_yoqeyNRV4IM", "preview"),
            ("AIzaSyBXF5wdsPOVwtEFWVh87Wi6CmTeFcNe-Ds", "experimental"),
            ("AIzaSyCveT194_XjazNc28mAIvqKMQolobrnED4", "experimental"),
            ("AIzaSyDUncd59xikStldixm0QeAsDPvJBxgB-5c", "experimental"),
            ("AIzaSyBossAMnU9krxaUw5F3moUJAXVgRNOJ0Uo", "experimental"),
            ("AIzaSyDyhQ_uxaDhf9kOEHEbyRWSseGfLqBHKl0", "experimental")
        ]
        
        self.key_trackers = [
            APIKeyTracker(key=key, key_type=key_type) 
            for key, key_type in api_keys
        ]
        
        # Model configuration
        self.models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        self.current_model_index = 0
        
        # Timeout for Flash model (shorter for better reliability)
        self.timeout_ms = 30 * 1000  # 30 seconds timeout
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.request_queue = Queue()
        self.result_queue = Queue()
        
        self.logger.info(f"Initialized parallel API manager with {len(self.key_trackers)} keys")
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.models[self.current_model_index % len(self.models)]
    
    def get_available_key(self) -> Optional[APIKeyTracker]:
        """Get the next available API key"""
        available_keys = [
            tracker for tracker in self.key_trackers 
            if tracker.is_available and tracker.can_make_request()
        ]
        
        if available_keys:
            # Return the key that's been unused the longest
            return min(available_keys, key=lambda k: k.last_request_time or datetime.min)
        
        return None
    
    def get_next_available_time(self) -> datetime:
        """Get when the next key will be available"""
        return min(tracker.get_next_available_time() for tracker in self.key_trackers)
    
    async def generate_content_single(self, prompt: str, key_tracker: APIKeyTracker) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content using a single API key with precise tracking
        """
        start_time = time.time()
        
        try:
            # Configure the client with longer timeout for legal document analysis
            client = genai.Client(
                api_key=key_tracker.key,
                http_options=HttpOptions(
                    timeout=60  # 60 second timeout for complex legal analysis
                )
            )
            
            # Make the API call
            response = client.models.generate_content(
                model=self.get_current_model(),
                contents=prompt
            )
            
            # Extract text response
            response_text = response.text or "No response received"
            
            # Record successful request
            tokens_used = len(prompt.split()) * 1.3  # Rough estimate
            key_tracker.record_request(int(tokens_used))
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                'key_type': key_tracker.key_type,
                'processing_time': processing_time,
                'model': self.get_current_model(),
                'tokens_used': tokens_used,
                'key_id': key_tracker.key[:10] + "..."
            }
            
            return response_text, metadata
            
        except Exception as e:
            key_tracker.record_error()
            processing_time = time.time() - start_time
            
            error_metadata = {
                'key_type': key_tracker.key_type,
                'processing_time': processing_time,
                'model': self.get_current_model(),
                'error': str(e),
                'key_id': key_tracker.key[:10] + "..."
            }
            
            self.logger.error(f"API call failed with {key_tracker.key_type} key: {e}")
            raise Exception(f"API call failed: {e}")
    
    async def generate_content_parallel(self, prompts: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate content for multiple prompts in parallel with smart key distribution
        """
        if not prompts:
            return []
        
        self.logger.info(f"Starting parallel generation for {len(prompts)} prompts")
        
        tasks = []
        results = []
        used_keys = set()  # Track which keys we've used to avoid conflicts
        
        for i, prompt in enumerate(prompts):
            # Find an available key (prefer unused keys)
            key_tracker = None
            attempts = 0
            
            while key_tracker is None and attempts < 30:  # Max 30 seconds wait
                # First, try to get an unused available key
                available_keys = [
                    tracker for tracker in self.key_trackers 
                    if tracker.is_available and tracker.can_make_request() 
                    and tracker.key not in used_keys
                ]
                
                if available_keys:
                    # Get the key that's been unused the longest
                    key_tracker = min(available_keys, key=lambda k: k.last_request_time or datetime.min)
                    used_keys.add(key_tracker.key)
                    break
                
                # If no unused keys, try any available key
                available_keys = [
                    tracker for tracker in self.key_trackers 
                    if tracker.is_available and tracker.can_make_request()
                ]
                
                if available_keys:
                    key_tracker = min(available_keys, key=lambda k: k.last_request_time or datetime.min)
                    break
                
                # Wait for next available key
                next_available = self.get_next_available_time()
                wait_time = min(1, (next_available - datetime.now()).total_seconds())
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                attempts += 1
            
            if key_tracker is None:
                raise Exception(f"No available keys after 30 seconds for prompt {i+1}")
            
            # Create task for this prompt
            task = asyncio.create_task(
                self.generate_content_single(prompt, key_tracker)
            )
            tasks.append((i, task))
            
            self.logger.info(f"Started task {i+1}/{len(prompts)} using {key_tracker.key_type} key {key_tracker.key[:10]}...")
            
            # Small delay to ensure keys don't conflict
            await asyncio.sleep(0.1)
        
        # Wait for all tasks to complete
        for i, task in tasks:
            try:
                result = await task
                results.append((i, result))
                self.logger.info(f"Completed task {i+1}")
            except Exception as e:
                self.logger.error(f"Task {i+1} failed: {e}")
                results.append((i, (f"Error: {e}", {"error": str(e)})))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    async def generate_content(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Single content generation method for compatibility with existing code
        Returns tuple of (response_text, metadata) to match expected interface
        """
        max_wait_attempts = 10  # Maximum number of wait attempts
        attempt = 0
        
        while attempt < max_wait_attempts:
            try:
                # Try to get an available key
                key_tracker = self.get_available_key()
                
                if key_tracker is not None:
                    # We have an available key, use it
                    response, metadata = await self.generate_content_single(prompt, key_tracker)
                    self.logger.info(f"Generated content using {metadata['key_type']} key in {metadata['processing_time']:.2f}s")
                    return response, metadata
                
                # No keys available, calculate wait time
                next_available = self.get_next_available_time()
                wait_time = (next_available - datetime.now()).total_seconds()
                
                if wait_time <= 0:
                    # Should be available now, try again immediately
                    attempt += 1
                    continue
                
                # Wait for the next available key
                self.logger.info(f"All keys rate-limited. Waiting {wait_time:.1f}s for next available key (attempt {attempt+1}/{max_wait_attempts})")
                await asyncio.sleep(min(wait_time + 1, 60))  # Cap wait time at 60 seconds
                attempt += 1
                
            except Exception as e:
                self.logger.error(f"Content generation attempt {attempt+1} failed: {e}")
                # Try with different model if current one fails
                if self.current_model_index < len(self.models) - 1:
                    self.current_model_index += 1
                    self.logger.info(f"Switching to model: {self.get_current_model()}")
                    return await self.generate_content(prompt, **kwargs)
                else:
                    attempt += 1
                    if attempt >= max_wait_attempts:
                        raise Exception(f"All Gemini API attempts failed: {e}")
                    
                    # Wait a bit before retrying
                    await asyncio.sleep(5)
        
        raise Exception("No API keys available after waiting - all keys are rate limited")
    
    def generate_content_sync(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous wrapper for generate_content with proper rate limiting
        Returns tuple of (response_text, metadata)
        """
        import time
        
        # Check if we have any available keys
        available_key = self.get_available_key()
        if available_key is None:
            # No keys available, wait and try to find any key
            self.logger.warning("All keys rate limited, waiting 10 seconds...")
            time.sleep(10)
            
            # Force use of the least recently used key
            available_key = min(self.key_trackers, key=lambda k: k.last_request_time or datetime.min)
            if not available_key.can_make_request():
                self.logger.warning("Still rate limited, using oldest key anyway...")
        
        try:
            # Run the async method in a new event loop if needed
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.generate_content(prompt, **kwargs))
                        return future.result(timeout=120)  # 2 minute timeout
                else:
                    return loop.run_until_complete(self.generate_content(prompt, **kwargs))
            except RuntimeError:
                # No event loop, create a new one
                return asyncio.run(self.generate_content(prompt, **kwargs))
        except Exception as e:
            self.logger.error(f"Sync content generation failed: {e}")
            raise e
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connectivity with detailed status
        """
        try:
            async def run_test():
                available_key = self.get_available_key()
                if available_key is None:
                    return {
                        "status": "no_keys_available",
                        "message": "No API keys available for testing",
                        "key_status": self.get_key_status()
                    }
                
                try:
                    response, metadata = await self.generate_content_single("Test connection", available_key)
                    return {
                        "status": "success",
                        "message": f"Connection successful using {metadata['key_type']} key",
                        "response_preview": response[:100] + "..." if len(response) > 100 else response,
                        "metadata": metadata,
                        "key_status": self.get_key_status()
                    }
                except Exception as e:
                    return {
                        "status": "connection_failed",
                        "message": f"Connection test failed: {e}",
                        "key_status": self.get_key_status()
                    }
            
            return asyncio.run(run_test())
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Test connection failed: {e}",
                "key_status": self.get_key_status()
            }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        """
        return self.models.copy()
    
    def set_model(self, model_name: str) -> bool:
        """
        Set the current model
        """
        if model_name in self.models:
            self.current_model_index = self.models.index(model_name)
            self.logger.info(f"Switched to model: {model_name}")
            return True
        else:
            self.logger.warning(f"Model {model_name} not available. Available models: {self.models}")
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get current API status including key availability
        """
        available_keys = sum(1 for tracker in self.key_trackers if tracker.is_available and tracker.can_make_request())
        total_keys = len(self.key_trackers)
        
        return {
            "working_keys": available_keys,
            "total_keys": total_keys,
            "current_model": self.get_current_model(),
            "status": "ready" if available_keys > 0 else "rate_limited",
            "key_status": self.get_key_status()
        }
    
    def get_key_status(self) -> List[Dict[str, Any]]:
        """
        Get detailed status of all API keys
        """
        key_status = []
        for i, tracker in enumerate(self.key_trackers):
            available = tracker.is_available and tracker.can_make_request()
            next_available = tracker.get_next_available_time()
            wait_time = (next_available - datetime.now()).total_seconds() if not available else 0
            
            key_status.append({
                "key_index": i,
                "key_type": tracker.key_type,
                "available": available,
                "requests_this_minute": len(tracker.requests_this_minute),
                "requests_today": len(tracker.requests_today),
                "error_count": tracker.error_count,
                "wait_time_seconds": max(0, wait_time),
                "last_used": tracker.last_request_time.isoformat() if tracker.last_request_time else None
            })
        
        return key_status

# Create global instance
parallel_api_manager = ParallelGeminiAPIManager()
