import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class APIKey:
    """Represents a Gemini API key with its metadata"""
    key: str
    key_type: str  # 'preview' or 'experimental'
    has_credit: bool
    last_used: Optional[datetime] = None
    error_count: int = 0
    rate_limit_reset: Optional[datetime] = None

class SmartAPIManager:
    """
    Enhanced API manager with smart key rotation for Gemini 2.5 Pro
    Supports preview keys (higher rate limits) and experimental keys (free but lower limits)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.preview_keys: List[APIKey] = []
        self.experimental_keys: List[APIKey] = []
        self.current_preview_index = 0
        self.current_experimental_index = 0
        
        # Model selection strategy based on key type
        self.preview_models = [
            "gemini-2.5-pro",  # Stable version
            "gemini-2.5-pro-preview-06-05",  # Preview version
            "gemini-2.5-pro-preview-05-06"   # Alternative preview
        ]
        self.experimental_model = "gemini-2.5-pro-exp-03-25"  # Experimental fallback
        self.current_model_index = 0
        
        # Safety settings for legal content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Rate limiting for experimental keys (2 requests per minute)
        self.experimental_rate_limit = 2  # requests per minute
        self.experimental_window = 60  # seconds
        self.experimental_request_times = []
        
        self._load_api_keys()
        self._check_key_credits()
        
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        # Load preview keys (higher rate limits)
        preview_keys = [
            os.getenv(f'GEMINI_PREVIEW_KEY_{i}') 
            for i in range(1, 6)  # Support up to 5 preview keys
        ]
        
        # Load experimental keys (free but lower limits)
        experimental_keys = [
            os.getenv(f'GEMINI_EXPERIMENTAL_KEY_{i}') 
            for i in range(1, 11)  # Support up to 10 experimental keys
        ]
        
        # Create APIKey objects
        for key in preview_keys:
            if key:
                self.preview_keys.append(APIKey(key=key, key_type='preview', has_credit=True))
        
        for key in experimental_keys:
            if key:
                self.experimental_keys.append(APIKey(key=key, key_type='experimental', has_credit=True))
        
        self.logger.info(f"Loaded {len(self.preview_keys)} preview keys and {len(self.experimental_keys)} experimental keys")
    
    def _check_key_credits(self):
        """Check which keys have available credits/quota"""
        async def check_key(api_key: APIKey):
            try:
                genai.configure(api_key=api_key.key)
                # Try a minimal test call
                model = genai.GenerativeModel(self.get_current_model(True))
                response = model.generate_content("Test", 
                                                generation_config=genai.types.GenerationConfig(
                                                    max_output_tokens=10,
                                                    temperature=0.0
                                                ))
                api_key.has_credit = True
                api_key.error_count = 0
                self.logger.info(f"Key {api_key.key[:10]}... has credit available")
            except Exception as e:
                if "quota" in str(e).lower() or "limit" in str(e).lower():
                    api_key.has_credit = False
                    self.logger.warning(f"Key {api_key.key[:10]}... has no credit or quota exceeded")
                else:
                    api_key.error_count += 1
                    self.logger.error(f"Error checking key {api_key.key[:10]}...: {e}")
        
        # Check all keys (simplified synchronous version for now)
        for key in self.preview_keys + self.experimental_keys:
            try:
                genai.configure(api_key=key.key)
                is_preview = key in self.preview_keys
                model = genai.GenerativeModel(self.get_current_model(is_preview))
                response = model.generate_content("Test", 
                                                generation_config=genai.types.GenerationConfig(
                                                    max_output_tokens=5,
                                                    temperature=0.0
                                                ))
                key.has_credit = True
                self.logger.debug(f"Key {key.key[:10]}... verified with model {self.get_current_model(is_preview)}")
            except Exception as e:
                if "quota" in str(e).lower() or "limit" in str(e).lower():
                    key.has_credit = False
                    self.logger.warning(f"Key {key.key[:10]}... has quota issues")
                elif "model" in str(e).lower() or "not found" in str(e).lower():
                    self.logger.warning(f"Model issue with key {key.key[:10]}...: {e}")
                    if is_preview:
                        self.rotate_model()
    
    def get_next_available_key(self, prefer_preview: bool = True) -> Optional[APIKey]:
        """Get the next available API key with smart rotation"""
        
        if prefer_preview and self.preview_keys:
            # Try preview keys first
            available_preview = [k for k in self.preview_keys if k.has_credit and k.error_count < 3]
            if available_preview:
                # Round-robin through available preview keys
                key = available_preview[self.current_preview_index % len(available_preview)]
                self.current_preview_index += 1
                return key
        
        # Fallback to experimental keys
        available_experimental = [k for k in self.experimental_keys if k.has_credit and k.error_count < 3]
        if available_experimental:
            key = available_experimental[self.current_experimental_index % len(available_experimental)]
            self.current_experimental_index += 1
            return key
        
        # If no keys available, try to reset error counts
        self.logger.warning("No available keys, resetting error counts")
        for key in self.preview_keys + self.experimental_keys:
            key.error_count = 0
        
        # Try once more
        if self.preview_keys:
            return self.preview_keys[0]
        elif self.experimental_keys:
            return self.experimental_keys[0]
        
        return None
    
    def generate_content(self, prompt: str, max_output_tokens: int = 2048, 
                        temperature: float = 0.7, prefer_preview: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content with smart key rotation and error handling
        Returns: (response_text, metadata)
        """
        max_retries = 3
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            api_key = self.get_next_available_key(prefer_preview)
            
            if not api_key:
                raise Exception("No available API keys")
            
            try:
                # Configure the API key
                genai.configure(api_key=api_key.key)
                
                # Determine if this is a preview key
                is_preview_key = api_key in self.preview_keys
                current_model = self.get_current_model(is_preview_key)
                
                # Create model with safety settings
                model = genai.GenerativeModel(
                    current_model,
                    safety_settings=self.safety_settings
                )
                
                # Generation config
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=0.8,
                    top_k=40
                )
                
                # Generate content
                start_time = time.time()
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                processing_time = time.time() - start_time
                
                # Update key metadata
                api_key.last_used = datetime.now()
                api_key.error_count = 0
                
                # Extract response text
                response_text = response.text if response.text else ""
                
                # Metadata
                metadata = {
                    'key_type': api_key.key_type,
                    'key_id': api_key.key[:10] + "...",
                    'processing_time': processing_time,
                    'model_used': current_model,
                    'attempt': attempt + 1,
                    'tokens_used': len(prompt.split()) + len(response_text.split())  # Rough estimate
                }
                
                self.logger.info(f"Successfully generated content using {api_key.key_type} key in {processing_time:.2f}s")
                return response_text, metadata
                
            except Exception as e:
                last_error = e
                attempt += 1
                api_key.error_count += 1
                
                # Check if it's a quota/credit issue
                error_str = str(e).lower()
                if "quota" in error_str or "limit" in error_str:
                    api_key.has_credit = False
                    self.logger.warning(f"Key {api_key.key[:10]}... hit quota limit")
                
                self.logger.error(f"Attempt {attempt} failed with key {api_key.key[:10]}...: {e}")
                
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"All attempts failed. Last error: {last_error}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all API keys for monitoring"""
        status = {
            'preview_keys': {
                'total': len(self.preview_keys),
                'available': len([k for k in self.preview_keys if k.has_credit and k.error_count < 3]),
                'with_credit': len([k for k in self.preview_keys if k.has_credit]),
                'keys': [
                    {
                        'id': k.key[:10] + "...",
                        'has_credit': k.has_credit,
                        'error_count': k.error_count,
                        'last_used': k.last_used.isoformat() if k.last_used else None
                    } for k in self.preview_keys
                ]
            },
            'experimental_keys': {
                'total': len(self.experimental_keys),
                'available': len([k for k in self.experimental_keys if k.has_credit and k.error_count < 3]),
                'with_credit': len([k for k in self.experimental_keys if k.has_credit]),
                'keys': [
                    {
                        'id': k.key[:10] + "...",
                        'has_credit': k.has_credit,
                        'error_count': k.error_count,
                        'last_used': k.last_used.isoformat() if k.last_used else None
                    } for k in self.experimental_keys
                ]
            }
        }
        return status
    
    def refresh_key_status(self):
        """Manually refresh the status of all keys"""
        self.logger.info("Refreshing API key status...")
        self._check_key_credits()
        self.logger.info("API key status refresh completed")

    def get_best_api_key(self) -> Optional[str]:
        """Get the best available API key, preferring preview keys"""
        # Try preview keys first
        for key in self.preview_keys:
            if key.has_credit and key.error_count < 3:
                return key.key
        
        # Fallback to experimental keys
        for key in self.experimental_keys:
            if key.has_credit and key.error_count < 3:
                return key.key
        
        return None

    def mark_key_error(self, api_key: str, error_type: str = 'general'):
        """Mark an API key as having an error"""
        for key in self.preview_keys + self.experimental_keys:
            if key.key == api_key:
                key.error_count += 1
                if error_type in ['quota_exceeded', 'billing']:
                    key.has_credit = False
                self.logger.warning(f"Marked key error for {key.key_type} key (errors: {key.error_count})")
                break
    
    def get_current_model(self, is_preview_key: bool = True) -> str:
        """Get the appropriate model based on key type and availability"""
        if is_preview_key:
            # Use preview models for preview keys
            return self.preview_models[self.current_model_index % len(self.preview_models)]
        else:
            # Use experimental model for experimental keys with rate limiting considerations
            return self.experimental_model
    
    def rotate_model(self):
        """Rotate to next available model on failures"""
        self.current_model_index = (self.current_model_index + 1) % len(self.preview_models)
        self.logger.info(f"Rotated to model: {self.get_current_model()}")
    
    def _can_use_experimental_key(self) -> bool:
        """Check if experimental key can be used based on rate limiting"""
        import time
        current_time = time.time()
        
        # Remove old requests outside the window
        self.experimental_request_times = [
            req_time for req_time in self.experimental_request_times 
            if current_time - req_time < self.experimental_window
        ]
        
        # Check if we're under the rate limit
        return len(self.experimental_request_times) < self.experimental_rate_limit
    
    def _record_experimental_request(self):
        """Record a request to experimental key for rate limiting"""
        import time
        self.experimental_request_times.append(time.time())
