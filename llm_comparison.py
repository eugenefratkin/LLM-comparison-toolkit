#!/usr/bin/env python3
"""
LLM Comparison Toolkit
Compares responses from Inception Labs API, OpenAI API, and Gemini AI
"""

import os
import time
import csv
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AI libraries
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed. Run: pip install openai")
    exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI library not installed. Run: pip install google-generativeai")
    exit(1)


class LLMComparison:
    def __init__(self):
        """Initialize the LLM comparison tool with API keys from environment variables."""
        self.inception_api_key = os.getenv('INCEPTION_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Validate API keys
        self._validate_api_keys()
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Inception Labs API endpoint (you may need to adjust this)
        self.inception_endpoint = "https://api.inceptionlabs.ai/v1/chat/completions"
    
    def _validate_api_keys(self):
        """Validate that all required API keys are present."""
        missing_keys = []
        
        if not self.inception_api_key:
            missing_keys.append('INCEPTION_API_KEY')
        if not self.openai_api_key:
            missing_keys.append('OPENAI_API_KEY')
        if not self.gemini_api_key:
            missing_keys.append('GEMINI_API_KEY')
        
        if missing_keys:
            print(f"Missing API keys: {', '.join(missing_keys)}")
            print("Please set these environment variables:")
            for key in missing_keys:
                print(f"  export {key}=your_api_key_here")
            exit(1)
    
    def read_prompt(self, filename: str = "prompt.txt") -> str:
        """Read the prompt from a text file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Prompt file '{filename}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            exit(1)
    
    def call_inception_api(self, prompt: str) -> Dict[str, Any]:
        """Call Inception Labs API and measure response time."""
        start_time = time.time()
        
        try:
            headers = {
                'Authorization': f'Bearer {self.inception_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": "gpt-3.5-turbo",  # Adjust model as needed
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.inception_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
                return {
                    'success': True,
                    'response': content,
                    'response_time': response_time,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'response': f"API Error: {response.status_code}",
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'response_time': response_time,
                'error': str(e)
            }
    
    def call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API and measure response time."""
        start_time = time.time()
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.choices[0].message.content
            return {
                'success': True,
                'response': content,
                'response_time': response_time,
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'response_time': response_time,
                'error': str(e)
            }
    
    def call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini AI API and measure response time."""
        start_time = time.time()
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.text if response.text else "No response generated"
            return {
                'success': True,
                'response': content,
                'response_time': response_time,
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'response_time': response_time,
                'error': str(e)
            }
    
    def run_comparison(self, output_file: str = "llm_comparison_results.csv"):
        """Run the comparison and save results to CSV."""
        print("Starting LLM Comparison...")
        
        # Read the prompt
        prompt = self.read_prompt()
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Call all APIs
        print("\nCalling APIs...")
        
        print("- Calling Inception Labs API...")
        inception_result = self.call_inception_api(prompt)
        
        print("- Calling OpenAI API...")
        openai_result = self.call_openai_api(prompt)
        
        print("- Calling Gemini AI...")
        gemini_result = self.call_gemini_api(prompt)
        
        # Prepare CSV data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write results to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'Timestamp', 'Prompt', 
                'Inception_Response', 'Inception_Time_Seconds', 'Inception_Success',
                'OpenAI_Response', 'OpenAI_Time_Seconds', 'OpenAI_Success',
                'Gemini_Response', 'Gemini_Time_Seconds', 'Gemini_Success'
            ])
            
            # Write response data
            writer.writerow([
                timestamp, prompt,
                inception_result['response'], f"{inception_result['response_time']:.3f}", inception_result['success'],
                openai_result['response'], f"{openai_result['response_time']:.3f}", openai_result['success'],
                gemini_result['response'], f"{gemini_result['response_time']:.3f}", gemini_result['success']
            ])
            
            # Write timing summary row
            writer.writerow([
                f"{timestamp} - Timing Summary", "Response Times (seconds)",
                f"Inception: {inception_result['response_time']:.3f}s", "", "",
                f"OpenAI: {openai_result['response_time']:.3f}s", "", "",
                f"Gemini: {gemini_result['response_time']:.3f}s", "", ""
            ])
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"\nInception Labs ({inception_result['response_time']:.3f}s):")
        print(f"  Success: {inception_result['success']}")
        print(f"  Response: {inception_result['response'][:200]}{'...' if len(inception_result['response']) > 200 else ''}")
        
        print(f"\nOpenAI ({openai_result['response_time']:.3f}s):")
        print(f"  Success: {openai_result['success']}")
        print(f"  Response: {openai_result['response'][:200]}{'...' if len(openai_result['response']) > 200 else ''}")
        
        print(f"\nGemini AI ({gemini_result['response_time']:.3f}s):")
        print(f"  Success: {gemini_result['success']}")
        print(f"  Response: {gemini_result['response'][:200]}{'...' if len(gemini_result['response']) > 200 else ''}")


def main():
    """Main function to run the LLM comparison."""
    try:
        comparison = LLMComparison()
        comparison.run_comparison()
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
