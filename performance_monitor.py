import time
import pandas as pd
from typing import Dict, List

class PerformanceMonitor:
    """
    Collects and visualizes performance metrics
    """
    def __init__(self):
        self.current_metrics: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

    # Timing helpers
    def start(self, key: str):
        self.current_metrics[f"{key}_start"] = time.time()

    def stop(self, key: str):
        start_key = f"{key}_start"
        if start_key in self.current_metrics:
            elapsed = time.time() - self.current_metrics[start_key]
            self.current_metrics[key] = round(elapsed, 4)
            del self.current_metrics[start_key]

    # Counters
    def increment(self, key: str, value: int = 1):
        self.current_metrics[key] = self.current_metrics.get(key, 0) + value

    # Finalize metrics for one run
    def finalize(self):
        self.history.append(self.current_metrics.copy())
        return self.current_metrics

    # DataFrames for visualization
    def latest_df(self):
        return pd.DataFrame([self.current_metrics])

    def history_df(self):
        return pd.DataFrame(self.history)
