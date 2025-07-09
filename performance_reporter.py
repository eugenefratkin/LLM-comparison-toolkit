from performance_metrics import BenchmarkMetrics
from typing import Dict, Any


class PerformanceReporter:
    """Formats and displays comprehensive performance benchmark results."""
    
    @staticmethod
    def print_performance_report(metrics: BenchmarkMetrics, duration_s: float):
        """Print a detailed performance report with all major metrics."""
        print("=" * 60)
        print(" PERFORMANCE BENCHMARK RESULTS ")
        print("=" * 60)
        print(f"{'Successful requests:':<40} {metrics.completed:<10}")
        print(f"{'Benchmark duration (s):':<40} {duration_s:<10.2f}")
        print(f"{'Total input tokens:':<40} {metrics.total_input_tokens:<10}")
        print(f"{'Total output tokens:':<40} {metrics.total_output_tokens:<10}")
        print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
        print(f"{'Output throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
        print(f"{'Total token throughput (tok/s):':<40} {metrics.total_token_throughput:<10.2f}")
        
        print("\n" + "-" * 60)
        print(" TIME TO FIRST TOKEN (TTFT) ")
        print("-" * 60)
        print(f"{'Mean TTFT (ms):':<40} {metrics.mean_ttft_ms:<10.2f}")
        print(f"{'Median TTFT (ms):':<40} {metrics.median_ttft_ms:<10.2f}")
        print(f"{'Std TTFT (ms):':<40} {metrics.std_ttft_ms:<10.2f}")
        for p, value in metrics.percentiles_ttft_ms:
            print(f"{'P' + str(int(p)) + ' TTFT (ms):':<40} {value:<10.2f}")
        
        if metrics.mean_tpot_ms > 0:
            print("\n" + "-" * 60)
            print(" TIME PER OUTPUT TOKEN (TPOT) ")
            print("-" * 60)
            print(f"{'Mean TPOT (ms):':<40} {metrics.mean_tpot_ms:<10.2f}")
            print(f"{'Median TPOT (ms):':<40} {metrics.median_tpot_ms:<10.2f}")
            print(f"{'Std TPOT (ms):':<40} {metrics.std_tpot_ms:<10.2f}")
            for p, value in metrics.percentiles_tpot_ms:
                print(f"{'P' + str(int(p)) + ' TPOT (ms):':<40} {value:<10.2f}")
        
        print("\n" + "-" * 60)
        print(" END-TO-END LATENCY (E2EL) ")
        print("-" * 60)
        print(f"{'Mean E2EL (ms):':<40} {metrics.mean_e2el_ms:<10.2f}")
        print(f"{'Median E2EL (ms):':<40} {metrics.median_e2el_ms:<10.2f}")
        print(f"{'Std E2EL (ms):':<40} {metrics.std_e2el_ms:<10.2f}")
        for p, value in metrics.percentiles_e2el_ms:
            print(f"{'P' + str(int(p)) + ' E2EL (ms):':<40} {value:<10.2f}")
        
        print("=" * 60)

    @staticmethod
    def print_provider_performance_report(provider_metrics: Dict[str, tuple], duration_s: float):
        """Print performance report broken down by provider with model information."""
        print("=" * 80)
        print(" PERFORMANCE BENCHMARK RESULTS BY PROVIDER ")
        print("=" * 80)

        for provider, (metrics, model_name) in provider_metrics.items():
            provider_name = provider.replace('_', ' ').title()
            print(f"\n{'='*15} {provider_name} ({model_name}) {'='*15}")
            print(f"{'Successful requests:':<40} {metrics.completed:<10}")
            print(f"{'Total input tokens:':<40} {metrics.total_input_tokens:<10}")
            print(f"{'Total output tokens:':<40} {metrics.total_output_tokens:<10}")
            print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
            print(f"{'Output throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
            print(f"{'Total token throughput (tok/s):':<40} {metrics.total_token_throughput:<10.2f}")

            print(f"\n{'Time to First Token (TTFT):':<40}")
            print(f"{'  Mean TTFT (ms):':<40} {metrics.mean_ttft_ms:<10.2f}")
            print(f"{'  Median TTFT (ms):':<40} {metrics.median_ttft_ms:<10.2f}")
            print(f"{'  P95 TTFT (ms):':<40} {metrics.percentiles_ttft_ms[2][1]:<10.2f}")

            print(f"\n{'End-to-End Latency (E2EL):':<40}")
            print(f"{'  Mean E2EL (ms):':<40} {metrics.mean_e2el_ms:<10.2f}")
            print(f"{'  Median E2EL (ms):':<40} {metrics.median_e2el_ms:<10.2f}")
            print(f"{'  P95 E2EL (ms):':<40} {metrics.percentiles_e2el_ms[2][1]:<10.2f}")

            if metrics.mean_tpot_ms > 0:
                print(f"\n{'Time per Output Token (TPOT):':<40}")
                print(f"{'  Mean TPOT (ms):':<40} {metrics.mean_tpot_ms:<10.2f}")
                print(f"{'  Median TPOT (ms):':<40} {metrics.median_tpot_ms:<10.2f}")
                print(f"{'  P95 TPOT (ms):':<40} {metrics.percentiles_tpot_ms[2][1]:<10.2f}")

        # Comparison summary
        print(f"\n{'='*20} PROVIDER COMPARISON {'='*20}")
        print(f"{'Provider (Model)':<30} {'Requests':<10} {'Avg E2EL (ms)':<15} {'Throughput (req/s)':<20}")
        print("-" * 85)
        for provider, (metrics, model_name) in provider_metrics.items():
            provider_name = provider.replace('_', ' ').title()
            provider_model = f"{provider_name} ({model_name})"
            print(f"{provider_model:<30} {metrics.completed:<10} {metrics.mean_e2el_ms:<15.2f} {metrics.request_throughput:<20.2f}")

        print("=" * 80)

    @staticmethod
    def print_summary_report(results: Dict[str, Any], metrics: BenchmarkMetrics = None):
        """Print a summary of comparison results with optional performance metrics."""
        print("\n" + "=" * 60)
        print(" LLM COMPARISON SUMMARY ")
        print("=" * 60)
        
        successful_providers = []
        failed_providers = []
        
        for provider, result in results.items():
            if result['success']:
                successful_providers.append(provider.upper())
            else:
                failed_providers.append(f"{provider.upper()}: {result['error']}")
        
        print(f"{'Successful providers:':<30} {', '.join(successful_providers) if successful_providers else 'None'}")
        print(f"{'Failed providers:':<30} {len(failed_providers)}")
        
        if failed_providers:
            print("\nFailure details:")
            for failure in failed_providers:
                print(f"  - {failure}")
        
        if metrics:
            print(f"\n{'Performance Summary:':<30}")
            print(f"{'  Total requests:':<30} {metrics.completed}")
            print(f"{'  Avg response time (ms):':<30} {metrics.mean_e2el_ms:.2f}")
            print(f"{'  Throughput (req/s):':<30} {metrics.request_throughput:.2f}")
        
        print("=" * 60)
