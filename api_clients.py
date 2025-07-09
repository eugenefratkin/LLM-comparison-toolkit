import time
import os
from typing import Dict, Any
from openai import OpenAI
import google.generativeai as genai
from performance_metrics import PerformanceTracker


class BaseAPIClient:
    """Base class for all API clients with common functionality."""
    
    def __init__(self, config: Dict[str, Any], performance_tracker: PerformanceTracker):
        self.config = config
        self.performance_tracker = performance_tracker
        
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """Call the API with the given prompt. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count based on word count."""
        return int(len(text.split()) * 1.3)


class InceptionAPIClient(BaseAPIClient):
    """API client for Inception Labs API."""
    
    def __init__(self, api_key: str, config: Dict[str, Any], performance_tracker: PerformanceTracker):
        super().__init__(config, performance_tracker)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.inceptionlabs.ai/v1"
        )
        
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """Call Inception Labs API with performance tracking."""
        start_time = self.performance_tracker.start_request()
        ttft_start = time.perf_counter()
        
        try:
            model_name = self.config['selected_models']['inception_labs']
            max_tokens = self.config['api_parameters']['max_tokens']
            temperature = self.config['api_parameters']['temperature']

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            content = ""
            ttft = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - ttft_start
                    content += chunk.choices[0].delta.content
            
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content)
            
            self.performance_tracker.record_metrics(start_time, ttft or 0, input_tokens, output_tokens)
            
            return {
                'success': True,
                'response': content,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }


class OpenAIAPIClient(BaseAPIClient):
    """API client for OpenAI API."""
    
    def __init__(self, api_key: str, config: Dict[str, Any], performance_tracker: PerformanceTracker):
        super().__init__(config, performance_tracker)
        self.client = OpenAI(api_key=api_key)
        
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API with performance tracking."""
        start_time = self.performance_tracker.start_request()
        ttft_start = time.perf_counter()
        
        try:
            model_name = self.config['selected_models']['openai']
            max_tokens = self.config['api_parameters']['max_tokens']
            temperature = self.config['api_parameters']['temperature']

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            content = ""
            ttft = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - ttft_start
                    content += chunk.choices[0].delta.content
            
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content)
            
            self.performance_tracker.record_metrics(start_time, ttft or 0, input_tokens, output_tokens)
            
            return {
                'success': True,
                'response': content,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }


class GeminiAPIClient(BaseAPIClient):
    """API client for Google Gemini API."""
    
    def __init__(self, api_key: str, config: Dict[str, Any], performance_tracker: PerformanceTracker):
        super().__init__(config, performance_tracker)
        genai.configure(api_key=api_key)
        gemini_model_name = self.config['selected_models']['gemini']
        self.model = genai.GenerativeModel(gemini_model_name)
        
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini API with performance tracking."""
        start_time = self.performance_tracker.start_request()
        ttft_start = time.perf_counter()
        
        try:
            response = self.model.generate_content(
                prompt,
                stream=True
            )
            
            content = ""
            ttft = None
            for chunk in response:
                if chunk.text:
                    if ttft is None:
                        ttft = time.perf_counter() - ttft_start
                    content += chunk.text
            
            if not content:
                content = "No response generated"
            
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(content)
            
            self.performance_tracker.record_metrics(start_time, ttft or 0, input_tokens, output_tokens)
            
            return {
                'success': True,
                'response': content,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }
