"""
Streamlit-Compatible Smart API Manager for Gemini API

This implementation addresses the persistent "No working API keys available" issue by:
1. Using Streamlit secrets management instead of runtime environment variables
2. Implementing real API connectivity checks with actual API calls
3. Providing robust error handling and debugging
4. Supporting multiple API key configurations and rotation
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import streamlit as st
from dataclasses import dataclass
from datetime import datetime

@dataclass
class APIKey:
    """Represents a Gemini API key with its metadata"""
    key: str
    key_id: str
    has_credit: bool = True
    last_used: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

class StreamlitAPIManager:
    """
    Streamlit-compatible API manager with robust connectivity testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_keys: List[APIKey] = []
        self.current_key_index = 0
        
        # Load API keys from Streamlit secrets
        self._load_api_keys()
        
        # Model configuration
        self.models = [
            "models/gemini-2.5-pro",
            "models/gemini-2.5-pro-exp-03-25",
            "models/gemini-1.5-pro"
        ]
        self.current_model = self.models[0]
        
        # Safety settings for legal content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def _load_api_keys(self):
        """Load API keys from Streamlit secrets or environment variables"""
        self.api_keys = []
        
        try:
            # Try loading from Streamlit secrets first
            if hasattr(st, 'secrets'):
                # Method 1: Multiple keys in gemini section
                if 'gemini' in st.secrets:
                    gemini_secrets = st.secrets['gemini']
                    for key_name, key_value in gemini_secrets.items():
                        if key_name.startswith('api_key') and key_value and len(key_value) > 20:
                            self.api_keys.append(APIKey(
                                key=key_value,
                                key_id=f"gemini_{key_name}"
                            ))
                
                # Method 2: Single key as GOOGLE_API_KEY
                if 'GOOGLE_API_KEY' in st.secrets and len(st.secrets['GOOGLE_API_KEY']) > 20:
                    self.api_keys.append(APIKey(
                        key=st.secrets['GOOGLE_API_KEY'],
                        key_id="google_api_key"
                    ))
                
                # Method 3: Single key as GEMINI_API_KEY  
                if 'GEMINI_API_KEY' in st.secrets and len(st.secrets['GEMINI_API_KEY']) > 20:
                    self.api_keys.append(APIKey(
                        key=st.secrets['GEMINI_API_KEY'],
                        key_id="gemini_api_key"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Could not load from Streamlit secrets: {e}")
        
        # Fallback to environment variables if no secrets found
        if not self.api_keys:
            import os
            env_keys = [
                ('GOOGLE_API_KEY', 'google_api_key'),
                ('GEMINI_API_KEY', 'gemini_api_key'),
                ('GEMINI_API_KEY_1', 'gemini_1'),
                ('GEMINI_API_KEY_2', 'gemini_2')
            ]
            
            for env_var, key_id in env_keys:
                key_value = os.getenv(env_var)
                if key_value and len(key_value) > 20:
                    self.api_keys.append(APIKey(
                        key=key_value,
                        key_id=key_id
                    ))
        
        self.logger.info(f"Loaded {len(self.api_keys)} API keys")
    
    def test_api_connectivity(self, api_key: APIKey) -> Tuple[bool, Optional[str]]:
        """
        Test actual API connectivity with a minimal request
        Returns (is_working, error_message)
        """
        try:
            # Configure the API with this specific key
            genai.configure(api_key=api_key.key)
            
            # Test 1: List available models (lightweight call)
            try:
                models = list(genai.list_models())
                if not models:
                    return False, "No models available"
            except Exception as e:
                if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                    return False, f"Authentication failed: {e}"
                elif "quota" in str(e).lower() or "limit" in str(e).lower():
                    return False, f"Quota exceeded: {e}"
                else:
                    # Try a more direct test
                    pass
            
            # Test 2: Minimal content generation
            try:
                model = genai.GenerativeModel(
                    model_name=self.current_model,
                    safety_settings=self.safety_settings
                )
                
                response = model.generate_content(
                    "Hello",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1,
                        temperature=0.0
                    )
                )
                
                # Check if response was generated or blocked
                if response.candidates and response.candidates[0].content:
                    return True, None
                elif response.candidates and response.candidates[0].finish_reason:
                    # Content was generated but possibly filtered
                    return True, None
                else:
                    return False, "No response generated"
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit" in error_msg:
                    return False, f"Quota/rate limit: {e}"
                elif "authentication" in error_msg or "permission" in error_msg:
                    return False, f"Authentication error: {e}"
                elif "model" in error_msg and "not found" in error_msg:
                    # Try with fallback model
                    try:
                        fallback_model = genai.GenerativeModel("models/gemini-1.5-pro")
                        response = fallback_model.generate_content("Test")
                        return True, "Working with fallback model"
                    except:
                        return False, f"Model not available: {e}"
                else:
                    return False, f"Generation error: {e}"
                    
        except Exception as e:
            return False, f"Configuration error: {e}"
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get comprehensive API status with real connectivity testing
        Returns status in format expected by enhanced_phase2_interface.py
        """
        if not self.api_keys:
            return {
                "gemini": {
                    "status": "inactive",
                    "error": "No API keys configured",
                    "working": False
                }
            }
        
        status = {}
        working_count = 0
        
        for i, api_key in enumerate(self.api_keys):
            provider_name = f"gemini_{i+1}" if len(self.api_keys) > 1 else "gemini"
            
            # Test connectivity
            is_working, error_msg = self.test_api_connectivity(api_key)
            
            if is_working:
                working_count += 1
                api_key.has_credit = True
                api_key.error_count = 0
                api_key.last_error = None
                api_key.last_used = datetime.now()
                
                status[provider_name] = {
                    "status": "active",
                    "working": True,
                    "error": None,
                    "key_id": api_key.key_id,
                    "last_used": api_key.last_used.isoformat()
                }
            else:
                api_key.has_credit = False
                api_key.error_count += 1
                api_key.last_error = error_msg
                
                status[provider_name] = {
                    "status": "inactive", 
                    "working": False,
                    "error": error_msg,
                    "key_id": api_key.key_id,
                    "error_count": api_key.error_count
                }
        
        # Add summary
        status["_summary"] = {
            "total_keys": len(self.api_keys),
            "working_keys": working_count,
            "all_keys_working": working_count == len(self.api_keys),
            "has_working_keys": working_count > 0
        }
        
        return status
    
    def get_working_key(self) -> Optional[APIKey]:
        """Get the next available working API key"""
        if not self.api_keys:
            return None
        
        # Try keys starting from current index
        for _ in range(len(self.api_keys)):
            key = self.api_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            if key.has_credit and key.error_count < 3:
                # Quick retest if it's been a while since last use
                if key.last_used and (datetime.now() - key.last_used).seconds > 300:  # 5 minutes
                    is_working, _ = self.test_api_connectivity(key)
                    if not is_working:
                        continue
                
                return key
        
        return None
    
    def generate_content(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate content using the best available API key"""
        key = self.get_working_key()
        if not key:
            raise Exception("No working API keys available")
        
        try:
            genai.configure(api_key=key.key)
            model = genai.GenerativeModel(
                model_name=self.current_model,
                safety_settings=self.safety_settings
            )
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1
                )
            )
            
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content.parts[0].text
                key.last_used = datetime.now()
                return content
            else:
                raise Exception("No content generated or content was blocked")
                
        except Exception as e:
            key.error_count += 1
            key.last_error = str(e)
            if "quota" in str(e).lower():
                key.has_credit = False
            raise e

# Global instance for Streamlit
if 'api_manager' not in st.session_state:
    st.session_state.api_manager = StreamlitAPIManager()

def get_api_manager() -> StreamlitAPIManager:
    """Get the global API manager instance"""
    return st.session_state.api_manager
