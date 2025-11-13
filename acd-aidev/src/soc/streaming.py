from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd


class LiveDataStreamer:
    """
    Streams live battery data point by point, simulating real-time data arrival.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        speed_factor: float = 1.0,
        start_index: int = 0,
    ):
        """
        Args:
            data_df: DataFrame with live test data (must have time_s column)
            speed_factor: Speed multiplier (1.0 = real-time, 10.0 = 10x faster)
            start_index: Starting index in the dataframe
        """
        self.data_df = data_df.sort_values("time_s").reset_index(drop=True)
        self.speed_factor = speed_factor
        self.current_index = start_index
        self.start_time = None
        self.data_start_time = self.data_df.iloc[0]["time_s"] if len(self.data_df) > 0 else 0.0
        
    def start(self):
        """Start the streaming session."""
        self.start_time = time.time()
        self.data_start_time = self.data_df.iloc[self.current_index]["time_s"]
        
    def get_next(self) -> Optional[pd.Series]:
        """
        Get the next data point based on elapsed time.
        Returns None if stream is exhausted.
        """
        if self.current_index >= len(self.data_df):
            return None
            
        if self.start_time is None:
            self.start()
            
        elapsed_real = time.time() - self.start_time
        elapsed_data = elapsed_real * self.speed_factor
        
        target_data_time = self.data_start_time + elapsed_data
        
        # Find the next data point that should be emitted
        while self.current_index < len(self.data_df):
            row_time = self.data_df.iloc[self.current_index]["time_s"]
            if row_time <= target_data_time:
                row = self.data_df.iloc[self.current_index].copy()
                self.current_index += 1
                return row
            else:
                # Wait for next data point
                break
                
        return None
    
    def stream_all(self, callback: Optional[Callable[[pd.Series], None]] = None) -> Iterator[pd.Series]:
        """
        Generator that yields all data points in sequence.
        If callback is provided, it's called for each point.
        """
        self.start()
        while True:
            point = self.get_next()
            if point is None:
                break
            if callback:
                callback(point)
            yield point
    
    def reset(self, start_index: int = 0):
        """Reset stream to beginning or specified index."""
        self.current_index = start_index
        self.start_time = None
        if start_index < len(self.data_df):
            self.data_start_time = self.data_df.iloc[start_index]["time_s"]


class SimulatedLiveStreamer:
    """
    Simulates live data streaming with configurable update interval.
    Useful for Streamlit apps where we want to show data progressively.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        update_interval_ms: int = 1000,
    ):
        """
        Args:
            data_df: DataFrame with live test data
            update_interval_ms: Milliseconds between updates
        """
        self.data_df = data_df.sort_values("time_s").reset_index(drop=True)
        self.update_interval_ms = update_interval_ms
        self.current_index = 0
        
    def get_batch_up_to_now(self, max_points: Optional[int] = None) -> pd.DataFrame:
        """
        Returns all data points up to current_index.
        Useful for progressive plotting.
        """
        end_idx = self.current_index
        if max_points:
            start_idx = max(0, end_idx - max_points)
        else:
            start_idx = 0
        return self.data_df.iloc[start_idx:end_idx].copy()
    
    def advance(self, n: int = 1):
        """Advance the stream by n points."""
        self.current_index = min(self.current_index + n, len(self.data_df))
    
    def reset(self):
        """Reset to beginning."""
        self.current_index = 0
    
    def is_complete(self) -> bool:
        """Check if stream is exhausted."""
        return self.current_index >= len(self.data_df)
    
    def get_progress(self) -> float:
        """Get progress as fraction [0, 1]."""
        if len(self.data_df) == 0:
            return 0.0
        return min(1.0, self.current_index / len(self.data_df))

