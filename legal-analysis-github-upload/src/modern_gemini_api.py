#!/usr/bin/env python3
"""
Modern Gemini 2.5 Pro API Manager using the new google-genai SDK
Specifically designed for Streamlit with proper timeout and session management
"""
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai.types import HttpOptions
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from datetime import datetime

@dataclass
class APIKey:
    """Represents a Gemini API key with its metadata"""
    key: str
    key_type: str  # 'preview' or 'experimental'
    has_credit: bool = True
    last_used: Optional[datetime] = None
    error_count: int = 0
    last_test_time: Optional[datetime] = None

class ModernGeminiAPIManager:
    """
    Modern API manager for Gemini 2.5 Pro using google-genai SDK
    Optimized for Streamlit with proper timeout handling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Gemini 2.5 Flash requires shorter timeouts (it's optimized for speed)
        self.timeout_ms = 2 * 60 * 1000  # 2 minutes for Flash model
        
        # Available models in order of preference (FREE TIER COMPATIBLE)
        self.models = [
            "gemini-2.5-flash",  # FREE TIER: 10 RPM, 250K TPM, 500 RPD
            "gemini-2.0-flash",  # FREE TIER: 15 RPM, 1M TPM, 1500 RPD  
            "gemini-1.5-flash",  # FREE TIER: 15 RPM, 250K TPM, 500 RPD
            "gemini-2.5-flash-lite-preview-06-17"  # FREE TIER: 15 RPM, 250K TPM, 500 RPD
        ]
        
        self.current_model_index = 0
        
        # API keys
        self.api_keys = [
            APIKey("AIzaSyBn_mF1-j6rckpYdEqtKc4VzvO53PkRJHk", "preview"),
            APIKey("AIzaSyDvSLlG56zNrCGWKv-lL3pCYlp5CO47jog", "preview"),
            APIKey("AIzaSyB1bc8KJ_BBbsEoKyndiABIFwKvV6KhM6Y", "preview"),
            APIKey("AIzaSyAamPAkIms_pl87n5jsAHojtixLJ1teGaU", "preview"),
            APIKey("AIzaSyCPz_EiSt4-1jFWvRrSS95_yoqeyNRV4IM", "preview"),
            APIKey("AIzaSyBXF5wdsPOVwtEFWVh87Wi6CmTeFcNe-Ds", "experimental"),
            APIKey("AIzaSyCveT194_XjazNc28mAIvqKMQolobrnED4", "experimental"),
            APIKey("AIzaSyDUncd59xikStldixm0QeAsDPvJBxgB-5c", "experimental"),
            APIKey("AIzaSyBossAMnU9krxaUw5F3moUJAXVgRNOJ0Uo", "experimental"),
            APIKey("AIzaSyDyhQ_uxaDhf9kOEHEbyRWSseGfLqBHKl0", "experimental")
        ]
        
        self.current_key_index = 0
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the genai client with proper timeout"""
        if self.api_keys:
            api_key = self.api_keys[0].key
            self.client = genai.Client(
                api_key=api_key,
                http_options=HttpOptions(timeout=self.timeout_ms)
            )
            self.logger.info(f"Initialized modern GenAI client with {len(self.api_keys)} keys")
        else:
            raise Exception("No API keys available")
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.models[self.current_model_index % len(self.models)]
    
    def rotate_model(self):
        """Rotate to next model on failures"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        self.logger.info(f"Rotated to model: {self.get_current_model()}")
    
    def get_next_key(self) -> Optional[APIKey]:
        """Get next available API key with rate limiting"""
        available_keys = [k for k in self.api_keys if k.has_credit and k.error_count < 3]
        
        if not available_keys:
            # Reset error counts if all keys are exhausted
            for key in self.api_keys:
                key.error_count = 0
            available_keys = self.api_keys
            
        if available_keys:
            key = available_keys[self.current_key_index % len(available_keys)]
            self.current_key_index += 1
            return key
        
        return None
    
    def test_api_connectivity(self, quick_test: bool = False) -> Dict[str, Any]:
        """
        Test API connectivity with Gemini 2.5 Pro specific considerations
        """
        results = {
            'gemini': {'status': 'inactive', 'details': 'Not tested'},
            'openai': {'status': 'inactive', 'details': 'Not available'}  # For compatibility
        }
        
        # Quick test uses shorter timeout
        timeout = 30 if quick_test else 120  # 30s for quick, 2min for full
        
        try:
            # Test with first available key
            api_key = self.get_next_key()
            if not api_key:
                results['gemini']['details'] = 'No API keys available'
                return results
            
            # Create client for this test
            test_client = genai.Client(
                api_key=api_key.key,
                http_options=HttpOptions(timeout=timeout * 1000)
            )
            
            # Simple connectivity test
            test_prompt = "Respond with exactly: API_TEST_SUCCESS"
            
            response = test_client.models.generate_content(
                model=self.get_current_model(),
                contents=test_prompt
            )
            
            if response and hasattr(response, 'text'):
                api_key.has_credit = True
                api_key.error_count = 0
                api_key.last_test_time = datetime.now()
                
                results['gemini'] = {
                    'status': 'active',
                    'details': f'Connected with {api_key.key_type} key',
                    'model': self.get_current_model(),
                    'response': response.text[:50] + '...' if len(response.text) > 50 else response.text
                }
            else:
                results['gemini']['details'] = 'Empty response received'
                
        except Exception as e:
            error_msg = str(e).lower()
            if api_key:
                if 'quota' in error_msg or 'limit' in error_msg:
                    api_key.has_credit = False
                api_key.error_count += 1
            
            results['gemini'] = {
                'status': 'inactive',
                'details': f'Connection failed: {str(e)[:100]}'
            }
            
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60)
    )
    def generate_content(self, prompt: str, max_output_tokens: int = 2048, 
                        temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content with Gemini 2.5 Pro using modern SDK
        """
        api_key = self.get_next_key()
        if not api_key:
            raise Exception("No available API keys")
        
        try:
            # Create client for this request
            client = genai.Client(
                api_key=api_key.key,
                http_options=HttpOptions(timeout=self.timeout_ms)
            )
            
            start_time = time.time()
            
            # Generate content
            response = client.models.generate_content(
                model=self.get_current_model(),
                contents=prompt
            )
            
            processing_time = time.time() - start_time
            
            # Extract response safely
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            else:
                response_text = "[Empty or blocked response]"
            
            # Update key status
            api_key.last_used = datetime.now()
            api_key.error_count = 0
            
            metadata = {
                'key_type': api_key.key_type,
                'key_id': api_key.key[:10] + "...",
                'processing_time': processing_time,
                'model_used': self.get_current_model(),
                'sdk_version': 'google-genai'
            }
            
            self.logger.info(f"Generated content using {api_key.key_type} key in {processing_time:.2f}s")
            return response_text, metadata
            
        except Exception as e:
            api_key.error_count += 1
            self.logger.error(f"Generation failed with {api_key.key_type} key: {e}")
            raise e
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get API status in format expected by enhanced Phase 2 interface
        """
        # Test connectivity
        connectivity = self.test_api_connectivity(quick_test=True)
        
        return {
            'providers': {
                'gemini': connectivity['gemini'],
                'openai': connectivity['openai']
            },
            'total_keys': len(self.api_keys),
            'working_keys': len([k for k in self.api_keys if k.has_credit and k.error_count < 3]),
            'current_model': self.get_current_model(),
            'sdk_version': 'google-genai (modern)',
            'timeout_ms': self.timeout_ms
        }

# For backward compatibility, create an alias
SmartAPIManager = ModernGeminiAPIManager
