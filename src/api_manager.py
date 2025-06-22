"""
API Manager for Gemini API calls with key rotation and rate limiting.
Handles both embedding and generation models with automatic fallback.
Updated for Gemini 2.5 Pro with smart credit vs free tier usage tracking.
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from loguru import logger
import random

load_dotenv()


class ModelType(Enum):
    """Types of Gemini models."""
    GENERATION = "generation"
    EMBEDDING = "embedding"


@dataclass
class APIKey:
    """Represents an API key with its status and usage tracking."""
    key: str
    is_active: bool = True
    last_used: float = 0
    error_count: int = 0
    last_error_time: float = 0
    has_credits: bool = False  # True for $300 credit keys
    usage_count: int = 0      # Track total usage
    daily_usage: int = 0      # Track daily usage


class GeminiAPIManager:
    """Manages Gemini API calls with smart credit usage and rate limiting."""
    
    def __init__(self):
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        
        # Load model names - CORRECTED for Gemini 2.5 Pro
        self.generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-pro-preview-06-05")
        self.fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-pro-exp-03-25")
        self.flash_model = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash-preview-05-20")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        
        # Rate limiting configuration - Smart usage tracking for credits vs free tier
        self.rate_limits = {
            "gemini-2.5-pro-preview-06-05": {"rpm": 10, "tpm": 250000, "rpd": 500, "has_credits": True},
            "gemini-2.5-pro-exp-03-25": {"rpm": 5, "tpm": 250000, "tpd": 1000000, "rpd": 25, "has_credits": False},
            "gemini-2.5-flash-preview-05-20": {"rpm": 15, "tpm": 1000000, "rpd": 300, "has_credits": False}
        }
        
        # Track API usage for smart allocation
        self.request_times: Dict[str, List[float]] = {}
        self.daily_requests: Dict[str, int] = {}
        self.last_request_time: Dict[str, float] = {}
        
        # Smart usage tracking
        self.credit_key_usage = {}  # Track $300 credit key usage
        self.free_key_usage = {}   # Track free tier key usage
        
        # Minimum delay between requests (seconds)
        self.min_delay = float(os.getenv("RATE_LIMIT_DELAY", "2"))
        
        logger.info(f"Initialized API manager with {len(self.api_keys)} keys")
        logger.info(f"Using Gemini 2.5 Pro models: {self.generation_model} -> {self.fallback_model}")
    
    def _load_api_keys(self) -> List[APIKey]:
        """Load API keys from environment with smart credit detection."""
        keys_str = os.getenv("GEMINI_KEYS", "")
        if not keys_str:
            raise ValueError("No GEMINI_KEYS found in environment")
        
        key_list = [k.strip() for k in keys_str.split(",") if k.strip()]
        api_keys = []
        
        # First key is the $300 credit key as per your specification
        for i, key in enumerate(key_list):
            has_credits = (i == 0)  # First key has $300 credits
            api_keys.append(APIKey(
                key=key, 
                has_credits=has_credits,
                usage_count=0,
                daily_usage=0
            ))
            if has_credits:
                logger.info(f"Loaded $300 credit key: ...{key[-4:]}")
            else:
                logger.info(f"Loaded free tier key: ...{key[-4:]}")
        
        return api_keys
    
    def _get_smart_key_for_model(self, model: str) -> Optional[APIKey]:
        """Get the best key for a model based on credits and usage."""
        model_config = self.rate_limits.get(model, {})
        needs_credits = model_config.get("has_credits", False)
        
        # If model requires credits, prioritize credit keys
        if needs_credits:
            # Try credit keys first
            credit_keys = [k for k in self.api_keys if k.has_credits and k.is_active]
            if credit_keys:
                # Use least used credit key
                best_key = min(credit_keys, key=lambda k: k.usage_count)
                logger.debug(f"Using credit key for {model}: ...{best_key.key[-4:]} (usage: {best_key.usage_count})")
                return best_key
            
            # If no credit keys available, fall back to free tier but log it
            logger.warning(f"No credit keys available for {model}, falling back to free tier")
        
        # Use free tier keys (or fallback for credit models)
        free_keys = [k for k in self.api_keys if k.is_active]
        if free_keys:
            # Round robin through free keys
            best_key = min(free_keys, key=lambda k: k.usage_count)
            logger.debug(f"Using free tier key for {model}: ...{best_key.key[-4:]} (usage: {best_key.usage_count})")
            return best_key
        
        logger.error("No active API keys available")
        return None
    
    def _get_next_active_key(self) -> Optional[APIKey]:
        """Get the next active API key using smart allocation."""
        # Prioritize credit keys if available
        credit_keys = [k for k in self.api_keys if k.has_credits and k.is_active]
        if credit_keys:
            return min(credit_keys, key=lambda k: k.usage_count)
        
        # Fall back to free tier keys
        free_keys = [k for k in self.api_keys if k.is_active]
        if free_keys:
            return min(free_keys, key=lambda k: k.usage_count)
        
        # All keys exhausted, try to reactivate
        logger.warning("All API keys exhausted, attempting to reactivate")
        self._reactivate_keys()
        
        # Try one more time
        for key in self.api_keys:
            if key.is_active:
                return key
        
        return None
    
    def _reactivate_keys(self):
        """Reactivate keys that have been in cooldown."""
        current_time = time.time()
        for key in self.api_keys:
            if not key.is_active and current_time - key.last_error_time > 600:  # 10 min cooldown
                key.is_active = True
                key.error_count = 0
                logger.info(f"Reactivated API key ending in ...{key.key[-4:]}")
    
    async def _check_rate_limits(self, model: str, api_key: APIKey) -> bool:
        """Check if we're within rate limits for the model and key."""
        if model not in self.rate_limits:
            return True
        
        limits = self.rate_limits[model]
        current_time = time.time()
        
        # Initialize tracking for this model if needed
        key_id = f"{model}_{api_key.key[-4:]}"
        if key_id not in self.request_times:
            self.request_times[key_id] = []
            self.daily_requests[key_id] = 0
            self.last_request_time[key_id] = 0
        
        # Remove old request times (older than 1 minute)
        self.request_times[key_id] = [t for t in self.request_times[key_id] if current_time - t < 60]
        
        # Check requests per minute
        if "rpm" in limits and len(self.request_times[key_id]) >= limits["rpm"]:
            wait_time = 60 - (current_time - self.request_times[key_id][0])
            logger.warning(f"Rate limit approaching for {model} on key ...{api_key.key[-4:]}, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            return await self._check_rate_limits(model, api_key)  # Recursive check
        
        # Check daily requests
        if "rpd" in limits and self.daily_requests[key_id] >= limits["rpd"]:
            logger.error(f"Daily request limit reached for {model} on key ...{api_key.key[-4:]}")
            return False
        
        # Enforce minimum delay between requests
        if self.last_request_time[key_id] > 0:
            elapsed = current_time - self.last_request_time[key_id]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        
        return True
    
    def _update_rate_limit_tracking(self, model: str, api_key: APIKey):
        """Update rate limit tracking after a successful request."""
        current_time = time.time()
        key_id = f"{model}_{api_key.key[-4:]}"
        
        if key_id not in self.request_times:
            self.request_times[key_id] = []
            self.daily_requests[key_id] = 0
        
        self.request_times[key_id].append(current_time)
        self.daily_requests[key_id] += 1
        self.last_request_time[key_id] = current_time
        
        # Update key usage tracking
        api_key.usage_count += 1
        api_key.daily_usage += 1
        api_key.last_used = current_time
        
        # Log usage for monitoring
        key_type = "CREDIT" if api_key.has_credits else "FREE"
        logger.info(f"[{key_type}] Used {model} on key ...{api_key.key[-4:]} (total: {api_key.usage_count}, daily: {api_key.daily_usage})")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_content(self, prompt: str, 
                             system_instruction: Optional[str] = None,
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None) -> Optional[str]:
        """Generate content using Gemini 2.5 Pro with smart key allocation."""
        # Smart model selection - try credit models first, then fallback
        models_to_try = [self.generation_model, self.fallback_model, self.flash_model]
        
        for model_name in models_to_try:
            # Get best key for this model
            api_key = self._get_smart_key_for_model(model_name)
            if not api_key:
                logger.warning(f"No suitable key for {model_name}, trying next model")
                continue
            
            # Check rate limits
            if not await self._check_rate_limits(model_name, api_key):
                logger.warning(f"Rate limit exceeded for {model_name} on key ...{api_key.key[-4:]}, trying next model")
                continue
            
            try:
                # Configure API with current key
                genai.configure(api_key=api_key.key)
                
                # Create model with system instruction if provided
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                }
                
                if max_tokens:
                    generation_config["max_output_tokens"] = max_tokens
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
                
                # Generate content
                key_type = "CREDIT" if api_key.has_credits else "FREE"
                logger.debug(f"[{key_type}] Generating content with {model_name} using key ...{api_key.key[-4:]}")
                response = await model.generate_content_async(prompt)
                
                # Update tracking
                self._update_rate_limit_tracking(model_name, api_key)
                api_key.error_count = 0  # Reset error count on success
                
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error with {model_name} using key ...{api_key.key[-4:]}: {error_msg}")
                
                # Update key status based on error
                api_key.error_count += 1
                api_key.last_error_time = time.time()
                
                if "quota" in error_msg.lower() or "429" in error_msg:
                    api_key.is_active = False
                    key_type = "CREDIT" if api_key.has_credits else "FREE"
                    logger.warning(f"[{key_type}] Deactivated key ...{api_key.key[-4:]} due to quota error")
                
                # Try next model
                continue
        
        logger.error("All models and keys exhausted")
        return None
    
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for multiple texts."""
        # Get next active key
        api_key = self._get_next_active_key()
        if not api_key:
            logger.error("No active API keys available for embeddings")
            return None
        
        try:
            # Configure API
            genai.configure(api_key=api_key.key)
            
            # Create embedding model
            model = genai.GenerativeModel(model_name=self.embedding_model)
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = 100  # Gemini embedding batch limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Generating embeddings for batch {i//batch_size + 1}")
                
                # Check rate limits
                if not await self._check_rate_limits(self.embedding_model, api_key):
                    await asyncio.sleep(30)  # Wait before retry
                
                result = await model.embed_content_async(batch)
                embeddings.extend(result['embedding'])
                
                # Update tracking
                self._update_rate_limit_tracking(self.embedding_model, api_key)
                
                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(self.min_delay)
            
            api_key.error_count = 0
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            api_key.error_count += 1
            api_key.last_error_time = time.time()
            
            if "quota" in str(e).lower():
                api_key.is_active = False
            
            return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and smart usage statistics."""
        active_keys = sum(1 for k in self.api_keys if k.is_active)
        credit_keys = [k for k in self.api_keys if k.has_credits]
        free_keys = [k for k in self.api_keys if not k.has_credits]
        
        status = {
            "total_keys": len(self.api_keys),
            "active_keys": active_keys,
            "credit_keys": len(credit_keys),
            "free_keys": len(free_keys),
            "models": {
                "generation": self.generation_model,
                "fallback": self.fallback_model,
                "flash": self.flash_model,
                "embedding": self.embedding_model
            },
            "daily_requests": dict(self.daily_requests),
            "credit_usage": {
                "total_usage": sum(k.usage_count for k in credit_keys),
                "daily_usage": sum(k.daily_usage for k in credit_keys),
                "active_credit_keys": sum(1 for k in credit_keys if k.is_active)
            },
            "free_usage": {
                "total_usage": sum(k.usage_count for k in free_keys),
                "daily_usage": sum(k.daily_usage for k in free_keys),
                "active_free_keys": sum(1 for k in free_keys if k.is_active)
            },
            "key_status": [
                {
                    "key": f"...{k.key[-4:]}",
                    "type": "CREDIT" if k.has_credits else "FREE",
                    "active": k.is_active,
                    "usage_count": k.usage_count,
                    "daily_usage": k.daily_usage,
                    "error_count": k.error_count,
                    "last_used": time.time() - k.last_used if k.last_used > 0 else None
                }
                for k in self.api_keys
            ]
        }
        
        return status


# Test function
async def test_api_manager():
    """Test the API manager functionality."""
    manager = GeminiAPIManager()
    
    # Test content generation
    prompt = "What are the key considerations for Gibraltar tax residency?"
    response = await manager.generate_content(
        prompt=prompt,
        system_instruction="You are a legal expert. Provide concise, accurate information.",
        temperature=0.3
    )
    
    if response:
        print(f"Generated response: {response[:200]}...")
    else:
        print("Failed to generate response")
    
    # Show API status
    status = manager.get_api_status()
    print(f"\nAPI Status: {status}")


if __name__ == "__main__":
    asyncio.run(test_api_manager())