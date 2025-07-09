#!/usr/bin/env python3
"""
LLM Comparison Toolkit
Compares responses from multiple Large Language Model APIs side by side with comprehensive performance metrics.
"""

import argparse
import time
from comparison_engine import LLMComparisonEngine
from performance_reporter import PerformanceReporter


def main():
    """Main function to run the LLM comparison tool."""
    parser = argparse.ArgumentParser(description="Compare LLM responses with performance benchmarking")
    parser.add_argument("--chain", action="store_true", help="Run prompt chain comparison")
    parser.add_argument("--models", action="store_true", help="List available models")
    parser.add_argument("--performance", action="store_true", help="Enable detailed performance reporting")
    
    args = parser.parse_args()
    
    # Initialize the comparison engine
    engine = LLMComparisonEngine()
    
    if args.models:
        engine.list_available_models()
        return
        
    if args.chain:
        start_time = time.perf_counter()
        engine.run_chain_comparison()
        duration = time.perf_counter() - start_time
        
        if args.performance:
            metrics = engine.calculate_benchmark_metrics(duration)
            if metrics:
                PerformanceReporter.print_performance_report(metrics, duration)
    else:
        start_time = time.perf_counter()
        prompt = engine.read_prompt()
        if prompt:
            results = engine.run_single_prompt_comparison(prompt)
            duration = time.perf_counter() - start_time
            
            # Save results to CSV
            engine.save_results_to_csv(results)
            
            # Display performance metrics if requested
            if args.performance:
                metrics = engine.calculate_benchmark_metrics(duration)
                if metrics:
                    PerformanceReporter.print_performance_report(metrics, duration)
                else:
                    print("No performance metrics available (no successful requests)")
            
            PerformanceReporter.print_summary_report(results, 
                engine.calculate_benchmark_metrics(duration) if args.performance else None)
        else:
            print("No prompt found. Please create a prompt.txt file or specify a prompt.")


if __name__ == "__main__":
    main()
