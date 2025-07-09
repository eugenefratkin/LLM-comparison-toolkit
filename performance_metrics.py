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
    """Tracks performance metrics during API calls."""
    
    def __init__(self):
        self.request_start_times = []
        self.ttfts = []
        self.e2els = []
        self.input_tokens = []
        self.output_tokens = []
        
    def start_request(self) -> float:
        """Start tracking a new request and return the start time."""
        start_time = time.perf_counter()
        self.request_start_times.append(start_time)
        return start_time
        
    def record_metrics(self, start_time: float, ttft: float, input_tokens: int, output_tokens: int):
        """Record performance metrics for a completed request."""
        e2el = time.perf_counter() - start_time
        self.ttfts.append(ttft * 1000)  # Convert to ms
        self.e2els.append(e2el * 1000)  # Convert to ms
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        
    def calculate_metrics(self, duration_s: float) -> Optional[BenchmarkMetrics]:
        """Calculate comprehensive benchmark metrics from collected data."""
        if not self.ttfts:
            return None
            
        ttfts = np.array(self.ttfts)
        e2els = np.array(self.e2els)
        total_input = sum(self.input_tokens)
        total_output = sum(self.output_tokens)
        completed = len(ttfts)
        
        tpots = []
        for i, e2el in enumerate(e2els):
            output_tokens = self.output_tokens[i]
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
