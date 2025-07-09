from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import time


@dataclass
class BenchmarkMetrics:
    """Comprehensive performance metrics for LLM benchmarking."""
    completed: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


class PerformanceTracker:
    """Tracks performance metrics during API calls per provider."""

    def __init__(self):
        # Track metrics per provider
        self.provider_metrics = {}
        # Track model names per provider
        self.provider_models = {}

    def _ensure_provider(self, provider: str):
        """Ensure provider tracking exists."""
        if provider not in self.provider_metrics:
            self.provider_metrics[provider] = {
                'request_start_times': [],
                'ttfts': [],
                'e2els': [],
                'input_tokens': [],
                'output_tokens': []
            }
        
    def start_request(self, provider: str, model_name: str = None) -> float:
        """Start tracking a new request for a specific provider and return the start time."""
        self._ensure_provider(provider)
        if model_name:
            self.provider_models[provider] = model_name
        start_time = time.perf_counter()
        self.provider_metrics[provider]['request_start_times'].append(start_time)
        return start_time

    def record_metrics(self, provider: str, start_time: float, ttft: float, input_tokens: int, output_tokens: int):
        """Record performance metrics for a completed request from a specific provider."""
        self._ensure_provider(provider)
        e2el = time.perf_counter() - start_time
        self.provider_metrics[provider]['ttfts'].append(ttft * 1000)  # Convert to ms
        self.provider_metrics[provider]['e2els'].append(e2el * 1000)  # Convert to ms
        self.provider_metrics[provider]['input_tokens'].append(input_tokens)
        self.provider_metrics[provider]['output_tokens'].append(output_tokens)
        
    def calculate_metrics_for_provider(self, provider: str, duration_s: float) -> Optional[BenchmarkMetrics]:
        """Calculate comprehensive benchmark metrics for a specific provider."""
        if provider not in self.provider_metrics:
            return None

        metrics = self.provider_metrics[provider]
        if not metrics['ttfts']:
            return None

        ttfts = np.array(metrics['ttfts'])
        e2els = np.array(metrics['e2els'])
        total_input = sum(metrics['input_tokens'])
        total_output = sum(metrics['output_tokens'])
        completed = len(ttfts)

        tpots = []
        for i, e2el in enumerate(e2els):
            output_tokens = metrics['output_tokens'][i]
            if output_tokens > 1:
                ttft = ttfts[i]
                tpot = (e2el - ttft) / (output_tokens - 1)
                tpots.append(tpot)

        tpots = np.array(tpots) if tpots else np.array([0])
        percentiles = [50, 90, 95, 99]

        return BenchmarkMetrics(
            completed=completed,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            request_throughput=completed / duration_s if duration_s > 0 else 0,
            output_throughput=total_output / duration_s if duration_s > 0 else 0,
            total_token_throughput=(total_input + total_output) / duration_s if duration_s > 0 else 0,
            mean_ttft_ms=float(np.mean(ttfts)),
            median_ttft_ms=float(np.median(ttfts)),
            std_ttft_ms=float(np.std(ttfts)),
            percentiles_ttft_ms=[(p, float(np.percentile(ttfts, p))) for p in percentiles],
            mean_tpot_ms=float(np.mean(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            median_tpot_ms=float(np.median(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            std_tpot_ms=float(np.std(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            percentiles_tpot_ms=[(p, float(np.percentile(tpots, p))) for p in percentiles] if len(tpots) > 0 and tpots[0] > 0 else [],
            mean_e2el_ms=float(np.mean(e2els)),
            median_e2el_ms=float(np.median(e2els)),
            std_e2el_ms=float(np.std(e2els)),
            percentiles_e2el_ms=[(p, float(np.percentile(e2els, p))) for p in percentiles]
        )

    def calculate_all_provider_metrics(self, duration_s: float) -> Dict[str, tuple]:
        """Calculate metrics for all providers with model information."""
        provider_metrics = {}
        for provider in self.provider_metrics.keys():
            metrics = self.calculate_metrics_for_provider(provider, duration_s)
            model_name = self.provider_models.get(provider, "Unknown")
            if metrics:
                provider_metrics[provider] = (metrics, model_name)
        return provider_metrics

    def calculate_metrics(self, duration_s: float) -> Optional[BenchmarkMetrics]:
        """Calculate aggregate benchmark metrics from all providers (for backward compatibility)."""
        # Aggregate all provider data
        all_ttfts = []
        all_e2els = []
        all_input_tokens = []
        all_output_tokens = []

        for provider_data in self.provider_metrics.values():
            all_ttfts.extend(provider_data['ttfts'])
            all_e2els.extend(provider_data['e2els'])
            all_input_tokens.extend(provider_data['input_tokens'])
            all_output_tokens.extend(provider_data['output_tokens'])

        if not all_ttfts:
            return None

        ttfts = np.array(all_ttfts)
        e2els = np.array(all_e2els)
        total_input = sum(all_input_tokens)
        total_output = sum(all_output_tokens)
        completed = len(ttfts)

        tpots = []
        for i, e2el in enumerate(e2els):
            output_tokens = all_output_tokens[i]
            if output_tokens > 1:
                ttft = ttfts[i]
                tpot = (e2el - ttft) / (output_tokens - 1)
                tpots.append(tpot)

        tpots = np.array(tpots) if tpots else np.array([0])
        percentiles = [50, 90, 95, 99]

        return BenchmarkMetrics(
            completed=completed,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            request_throughput=completed / duration_s if duration_s > 0 else 0,
            output_throughput=total_output / duration_s if duration_s > 0 else 0,
            total_token_throughput=(total_input + total_output) / duration_s if duration_s > 0 else 0,
            mean_ttft_ms=float(np.mean(ttfts)),
            median_ttft_ms=float(np.median(ttfts)),
            std_ttft_ms=float(np.std(ttfts)),
            percentiles_ttft_ms=[(p, float(np.percentile(ttfts, p))) for p in percentiles],
            mean_tpot_ms=float(np.mean(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            median_tpot_ms=float(np.median(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            std_tpot_ms=float(np.std(tpots)) if len(tpots) > 0 and tpots[0] > 0 else 0,
            percentiles_tpot_ms=[(p, float(np.percentile(tpots, p))) for p in percentiles] if len(tpots) > 0 and tpots[0] > 0 else [],
            mean_e2el_ms=float(np.mean(e2els)),
            median_e2el_ms=float(np.median(e2els)),
            std_e2el_ms=float(np.std(e2els)),
            percentiles_e2el_ms=[(p, float(np.percentile(e2els, p))) for p in percentiles]
        )
        
    def reset(self):
        """Reset all tracked metrics."""
        self.request_start_times.clear()
        self.ttfts.clear()
        self.e2els.clear()
        self.input_tokens.clear()
        self.output_tokens.clear()
