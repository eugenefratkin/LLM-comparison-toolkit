import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from performance_metrics import PerformanceTracker, BenchmarkMetrics
from api_clients import InceptionAPIClient, OpenAIAPIClient, GeminiAPIClient
from dotenv import load_dotenv


class LLMComparisonEngine:
    """Main engine for comparing LLM responses with comprehensive performance metrics."""
    
    def __init__(self, config_file: str = "models_config.json"):
        self.config = self._load_config(config_file)
        self.performance_tracker = PerformanceTracker()
        self.results = []
        self.timing_data = []
        # Load environment variables from .env file
        load_dotenv(override=True)
        self._initialize_clients()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error parsing {config_file}. Please check the JSON format.")
            return {}
            
    def _initialize_clients(self):
        """Initialize API clients with performance tracking for multiple models per provider."""
        self.clients = {}

        # Debug: Check if .env file exists
        if not os.path.exists('.env'):
            print("Warning: .env file not found. Please create one from .env.example")
        else:
            print("✓ Found .env file")

        # Initialize clients for each provider and model combination
        inception_key = os.getenv('INCEPTION_API_KEY')
        if inception_key:
            inception_models = self.config.get('selected_models', {}).get('inception_labs', [])
            if isinstance(inception_models, str):
                inception_models = [inception_models]  # Handle backward compatibility

            for model in inception_models:
                client_key = f'inception_labs_{model}'
                self.clients[client_key] = InceptionAPIClient(
                    inception_key, self.config, self.performance_tracker, model
                )
            print(f"✓ Inception Labs API clients initialized for {len(inception_models)} model(s): {', '.join(inception_models)}")
        else:
            print("⚠ Inception Labs API key not found (INCEPTION_API_KEY)")

        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            openai_models = self.config.get('selected_models', {}).get('openai', [])
            if isinstance(openai_models, str):
                openai_models = [openai_models]  # Handle backward compatibility

            for model in openai_models:
                client_key = f'openai_{model}'
                self.clients[client_key] = OpenAIAPIClient(
                    openai_key, self.config, self.performance_tracker, model
                )
            print(f"✓ OpenAI API clients initialized for {len(openai_models)} model(s): {', '.join(openai_models)}")
        else:
            print("⚠ OpenAI API key not found (OPENAI_API_KEY)")

        # Initialize Gemini clients
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            gemini_models = self.config.get('selected_models', {}).get('gemini', [])
            if isinstance(gemini_models, str):
                gemini_models = [gemini_models]  # Handle backward compatibility

            for model in gemini_models:
                client_key = f'gemini_{model}'
                self.clients[client_key] = GeminiAPIClient(
                    gemini_key, self.config, self.performance_tracker, model
                )
            print(f"✓ Gemini API clients initialized for {len(gemini_models)} model(s): {', '.join(gemini_models)}")
        else:
            print("⚠ Gemini API key not found (GEMINI_API_KEY)")

        if not self.clients:
            print("❌ No API clients initialized. Please check your .env file and API keys.")
            return

        print(f"📊 Initialized {len(self.clients)} total client(s): {', '.join(self.clients.keys())}")
    
    def list_available_models(self):
        """Display available models for each API provider."""
        print("\nAvailable Models:")
        print("=" * 50)

        skip_keys = {'selected_models', 'api_parameters'}

        for provider, provider_config in self.config.items():
            if provider in skip_keys or not isinstance(provider_config, dict):
                continue

            available_models = provider_config.get('available_models', [])
            selected_models = self.config.get('selected_models', {}).get(provider, [])
            if isinstance(selected_models, str):
                selected_models = [selected_models]  # Handle backward compatibility

            if available_models:
                print(f"\n{provider.upper()}:")
                for model in available_models:
                    marker = " ✓ (selected)" if model in selected_models else ""
                    print(f"  - {model}{marker}")

        # Show current selection summary
        print(f"\n{'='*20} CURRENT SELECTION {'='*20}")
        selected_models = self.config.get('selected_models', {})
        for provider, models in selected_models.items():
            if isinstance(models, str):
                models = [models]
            provider_name = provider.replace('_', ' ').title()
            print(f"{provider_name}: {', '.join(models)}")
        print("=" * 62)
                
    def read_prompt(self, prompt_file: str = "prompt.txt") -> str:
        """Read prompt from file."""
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Prompt file {prompt_file} not found.")
            return ""
            
    def run_single_prompt_comparison(self, prompt: str) -> Dict[str, Any]:
        """Run comparison across all provider-model combinations for a single prompt."""
        results = {}

        print(f"\nRunning comparison for prompt: {prompt[:100]}...")
        print("=" * 60)

        for client_key, client in self.clients.items():
            # Extract provider and model from client key (e.g., "openai_gpt-4o" -> "OpenAI (gpt-4o)")
            parts = client_key.split('_', 1)
            provider = parts[0].replace('_', ' ').title()
            model = parts[1] if len(parts) > 1 else "Unknown"
            display_name = f"{provider} ({model})"

            print(f"\nCalling {display_name} API...")
            result = client.call_api(prompt)
            results[client_key] = result

            if result['success']:
                print(f"✓ {display_name}: Success")
                print(f"Response: {result['response'][:200]}...")
            else:
                print(f"✗ {display_name}: Failed - {result['error']}")

        return results
        
    def run_chain_comparison(self, chain_file: str = "input_prompt_chain.json"):
        """Run comparison for a prompt chain."""
        try:
            with open(chain_file, 'r') as f:
                chain_data = json.load(f)
        except FileNotFoundError:
            print(f"Chain file {chain_file} not found.")
            return
            
        chain = chain_data.get('steps', [])
        if not chain:
            print("No steps found in the chain file.")
            return

        print(f"\nRunning chain comparison: {chain_data.get('chain_name', 'Unnamed Chain')}")
        print(f"Description: {chain_data.get('description', 'No description')}")
        print(f"Steps: {len(chain)}")
        print("=" * 60)
        print("=" * 60)
        
        for client_key, client in self.clients.items():
            # Extract provider and model from client key
            parts = client_key.split('_', 1)
            provider = parts[0].replace('_', ' ').title()
            model = parts[1] if len(parts) > 1 else "Unknown"
            display_name = f"{provider} ({model})"

            print(f"\n--- Executing chain for {display_name.upper()} ---")
            self._execute_chain_for_provider(client_key, client, chain, chain_data)
            
    def _execute_chain_for_provider(self, provider: str, client, chain: List[Dict], chain_data: Dict[str, Any]):
        """Execute a prompt chain for a specific provider."""
        outputs = {}
        
        for step in chain:
            step_id = step.get('step_id', 0)
            step_name = step.get('name', f'Step {step_id}')
            prompt = step.get('prompt', '')
            
            print(f"\nStep {step_id}: {step_name}")
            
            if step.get('use_previous_output', False) and outputs:
                previous_key = step.get('output_variable', f'step_{step_id-1}')
                if previous_key in outputs:
                    prompt = prompt.replace('{previous_output}', outputs[previous_key])
                    
            result = client.call_api(prompt)
            
            if result['success']:
                output_var = step.get('output_variable', f'step_{step_id}')
                outputs[output_var] = result['response']
                print(f"✓ Success: {result['response'][:100]}...")
            else:
                print(f"✗ Failed: {result['error']}")
                if not chain_data.get('chain_parameters', {}).get('continue_on_error', False):
                    break
                    
    def calculate_benchmark_metrics(self, duration_s: float) -> Optional[BenchmarkMetrics]:
        """Calculate comprehensive benchmark metrics from collected data."""
        return self.performance_tracker.calculate_metrics(duration_s)

    def calculate_provider_metrics(self, duration_s: float) -> Dict[str, tuple]:
        """Calculate benchmark metrics per provider with model information."""
        return self.performance_tracker.calculate_all_provider_metrics(duration_s)
        
    def save_results_to_csv(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save comparison results to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_comparison_results_{timestamp}.csv"
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['provider', 'success', 'response', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for provider, result in results.items():
                writer.writerow({
                    'provider': provider,
                    'success': result['success'],
                    'response': result['response'][:500] if result['response'] else '',
                    'error': result['error'] or ''
                })
                
        print(f"\nResults saved to {filename}")
        
    def reset_performance_tracking(self):
        """Reset performance tracking for new benchmark run."""
        self.performance_tracker.reset()
