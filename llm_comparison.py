#!/usr/bin/env python3
"""
LLM Comparison Toolkit
Compares responses from Inception Labs API, OpenAI API, and Gemini AI
"""

import os
import time
import csv
import json

from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AI libraries
try:
    from openai import OpenAI, ChatCompletion
except ImportError:
    print("OpenAI library not installed. Run: pip install openai")
    exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI library not installed. Run: pip install google-generativeai")
    exit(1)


class LLMComparison:
    def __init__(self, config_file: str = "models_config.json"):
        """Initialize the LLM comparison tool with API keys and model configuration."""
        self.inception_api_key = os.getenv('INCEPTION_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

        # Load model configuration
        self.config = self._load_config(config_file)

        # Validate API keys
        self._validate_api_keys()

        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.inception_client = OpenAI(
            api_key=self.inception_api_key,
            base_url="https://api.inceptionlabs.ai/v1"
        )

        # Initialize Gemini client
        genai.configure(api_key=self.gemini_api_key)

        # Initialize Gemini model with configured model name
        gemini_model_name = self.config['selected_models']['gemini']
        self.gemini_model = genai.GenerativeModel(gemini_model_name)

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = json.load(file)

            # Validate that selected models are available
            self._validate_model_selection(config)
            return config

        except FileNotFoundError:
            print(f"Configuration file '{config_file}' not found.")
            print("Please create the configuration file or run with default settings.")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            exit(1)

    def _validate_model_selection(self, config: Dict[str, Any]):
        """Validate that selected models are in the available models list."""
        for provider in ['inception_labs', 'openai', 'gemini']:
            selected_model = config['selected_models'][provider]
            available_models = config[provider]['available_models']

            if selected_model not in available_models:
                print(f"Error: Selected model '{selected_model}' for {provider} is not in available models.")
                print(f"Available models for {provider}: {', '.join(available_models)}")
                exit(1)

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

    def display_configuration(self):
        """Display current model configuration."""
        print("="*60)
        print("CURRENT MODEL CONFIGURATION")
        print("="*60)
        print(f"Inception Labs: {self.config['selected_models']['inception_labs']}")
        print(f"OpenAI: {self.config['selected_models']['openai']}")
        print(f"Gemini: {self.config['selected_models']['gemini']}")
        print(f"Max Tokens: {self.config['api_parameters']['max_tokens']}")
        print(f"Temperature: {self.config['api_parameters']['temperature']}")
        print("="*60)

    def list_available_models(self):
        """Display all available models for each provider."""
        print("\nAVAILABLE MODELS:")
        print("-" * 40)

        for provider in ['inception_labs', 'openai', 'gemini']:
            provider_name = provider.replace('_', ' ').title()
            print(f"\n{provider_name}:")
            models = self.config[provider]['available_models']
            selected = self.config['selected_models'][provider]

            for model in models:
                marker = " ✓ (selected)" if model == selected else "  "
                print(f"{marker} {model}")

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
            model_name = self.config['selected_models']['inception_labs']
            max_tokens = self.config['api_parameters']['max_tokens']
            temperature = self.config['api_parameters']['temperature']

            response = self.inception_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
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
    
    def call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API and measure response time."""
        start_time = time.time()

        try:
            model_name = self.config['selected_models']['openai']
            max_tokens = self.config['api_parameters']['max_tokens']
            temperature = self.config['api_parameters']['temperature']

            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
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

        # Display current configuration
        self.display_configuration()

        # Read the prompt
        prompt = self.read_prompt()
        print(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # Call all APIs
        print("\nCalling APIs...")
        
        inception_model = self.config['selected_models']['inception_labs']
        openai_model = self.config['selected_models']['openai']
        gemini_model = self.config['selected_models']['gemini']

        print(f"- Calling Inception Labs API ({inception_model})...")
        inception_result = self.call_inception_api(prompt)

        print(f"- Calling OpenAI API ({openai_model})...")
        openai_result = self.call_openai_api(prompt)

        print(f"- Calling Gemini AI ({gemini_model})...")
        gemini_result = self.call_gemini_api(prompt)
        
        # Prepare CSV data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write results to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'Timestamp', 'Prompt',
                f'Inception_Response_({inception_model})', 'Inception_Time_Seconds', 'Inception_Success',
                f'OpenAI_Response_({openai_model})', 'OpenAI_Time_Seconds', 'OpenAI_Success',
                f'Gemini_Response_({gemini_model})', 'Gemini_Time_Seconds', 'Gemini_Success'
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
        print(f"\nInception Labs - {inception_model} ({inception_result['response_time']:.3f}s):")
        print(f"  Success: {inception_result['success']}")
        print(f"  Response: {inception_result['response'][:200]}{'...' if len(inception_result['response']) > 200 else ''}")

        print(f"\nOpenAI - {openai_model} ({openai_result['response_time']:.3f}s):")
        print(f"  Success: {openai_result['success']}")
        print(f"  Response: {openai_result['response'][:200]}{'...' if len(openai_result['response']) > 200 else ''}")

        print(f"\nGemini AI - {gemini_model} ({gemini_result['response_time']:.3f}s):")
        print(f"  Success: {gemini_result['success']}")
        print(f"  Response: {gemini_result['response'][:200]}{'...' if len(gemini_result['response']) > 200 else ''}")


def main():
    """Main function to run the LLM comparison."""
    import sys

    try:
        comparison = LLMComparison()

        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] in ['--models', '-m', '--list-models']:
                comparison.list_available_models()
                print("\nTo change models, edit the 'selected_models' section in models_config.json")
                return
            elif sys.argv[1] in ['--help', '-h']:
                print("LLM Comparison Toolkit")
                print("Usage:")
                print("  python llm_comparison.py           # Run comparison")
                print("  python llm_comparison.py --models  # List available models")
                print("  python llm_comparison.py --help    # Show this help")
                return

        comparison.run_comparison()

    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
