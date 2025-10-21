"""Logging utilities for Phase 5 evaluation runs."""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, TextIO
from pathlib import Path


class RunLogger:
    """Logger for Phase 5 evaluation runs with timestamped output."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "phase5"):
        """
        Initialize run logger.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of experiment (used in log filename)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{experiment_name}_{timestamp}.txt"
        
        self.file_handle: Optional[TextIO] = None
        self.start_time = datetime.now()
        
        # Open log file
        self._open_log()
        
    def _open_log(self):
        """Open log file for writing."""
        try:
            self.file_handle = open(self.log_file, 'w', encoding='utf-8')
            self._write_header()
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file}: {e}")
            self.file_handle = None
    
    def _write_header(self):
        """Write log header with timestamp and system info."""
        if self.file_handle is None:
            return
            
        header = f"""
{'='*80}
Weight of Reasoning - Phase 5 Evaluation Run
{'='*80}
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Log File: {self.log_file}
{'='*80}

"""
        self.file_handle.write(header)
        self.file_handle.flush()
    
    def log(self, message: str, also_print: bool = True):
        """
        Log a message to file and optionally print to console.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        
        if self.file_handle is not None:
            self.file_handle.write(log_line)
            self.file_handle.flush()
        
        if also_print:
            print(message)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        self.log("\n=== Configuration ===")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
        self.log("")
    
    def log_hardware(self, hw_config: Dict[str, Any]):
        """
        Log hardware configuration.
        
        Args:
            hw_config: Hardware configuration dictionary
        """
        self.log("\n=== Hardware Configuration ===")
        self.log(f"  Device: {hw_config.get('device', 'unknown')}")
        self.log(f"  Device Name: {hw_config.get('device_name', 'N/A')}")
        self.log(f"  GPU Memory: {hw_config.get('gpu_memory_gb', 0):.1f} GB")
        self.log(f"  Dtype: {hw_config.get('dtype', 'unknown')}")
        self.log(f"  Batch Size: {hw_config.get('batch_size', 1)}")
        self.log("")
    
    def log_dataset_info(self, dataset_name: str, n_samples: int):
        """
        Log dataset information.
        
        Args:
            dataset_name: Name of dataset
            n_samples: Number of samples
        """
        self.log(f"  Loaded {dataset_name}: {n_samples} samples")
    
    def log_model_start(self, model_name: str, n_params: int):
        """
        Log start of model processing.
        
        Args:
            model_name: Name of model
            n_params: Number of parameters
        """
        self.log(f"\n{'='*80}")
        self.log(f"Processing Model: {model_name}")
        self.log(f"Parameters: {n_params:,}")
        self.log(f"{'='*80}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log metric values.
        
        Args:
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names
        """
        self.log(f"\n=== {prefix} Metrics ===" if prefix else "\n=== Metrics ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.log(f"  {metric}: {value:.4f}")
            else:
                self.log(f"  {metric}: {value}")
    
    def log_baseline_comparison(self, baseline_results: Dict[str, Any]):
        """
        Log baseline comparison results.
        
        Args:
            baseline_results: Baseline comparison dictionary
        """
        self.log("\n=== Baseline Comparison ===")
        self.log(f"  AUROC (baseline): {baseline_results.get('auroc_baseline', float('nan')):.4f}")
        self.log(f"  AUROC (REV): {baseline_results.get('auroc_rev', float('nan')):.4f}")
        self.log(f"  AUROC (combined): {baseline_results.get('auroc_combined', float('nan')):.4f}")
        self.log(f"  ΔAUC: {baseline_results.get('delta_auc', float('nan')):.4f}")
        self.log(f"  Samples: {baseline_results.get('n_samples', 0)}")
    
    def log_robustness(self, seed: int, temp: float, results: Dict[str, Any]):
        """
        Log robustness test results.
        
        Args:
            seed: Random seed used
            temp: Temperature used
            results: Results dictionary
        """
        self.log(f"\n=== Robustness Test (seed={seed}, temp={temp}) ===")
        for key, value in results.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
    
    def log_runtime(self, phase: str, runtime_sec: float):
        """
        Log runtime for a phase.
        
        Args:
            phase: Name of phase
            runtime_sec: Runtime in seconds
        """
        minutes = runtime_sec / 60
        self.log(f"  {phase} runtime: {runtime_sec:.1f}s ({minutes:.2f} min)")
    
    def log_success_criteria(self, criteria: Dict[str, bool]):
        """
        Log success criteria validation.
        
        Args:
            criteria: Dictionary of criterion name to pass/fail boolean
        """
        self.log("\n=== Success Criteria Validation ===")
        all_passed = all(criteria.values())
        
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.log(f"  {status}: {criterion}")
        
        self.log(f"\nOverall: {'✅ ALL CRITERIA PASSED' if all_passed else '❌ SOME CRITERIA FAILED'}")
    
    def finalize(self, total_runtime_sec: Optional[float] = None):
        """
        Finalize log with summary and close file.
        
        Args:
            total_runtime_sec: Total runtime in seconds (computed if not provided)
        """
        if total_runtime_sec is None:
            total_runtime_sec = (datetime.now() - self.start_time).total_seconds()
        
        if self.file_handle is not None:
            footer = f"""
{'='*80}
Evaluation Complete
Total Runtime: {total_runtime_sec:.1f}s ({total_runtime_sec/60:.2f} min)
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
            self.file_handle.write(footer)
            self.file_handle.close()
            self.file_handle = None
        
        self.log(f"\nLog saved to: {self.log_file}", also_print=True)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.log(f"\nERROR: {exc_type.__name__}: {exc_val}")
        self.finalize()
        return False


def create_run_logger(log_dir: str = "logs", experiment_name: str = "phase5") -> RunLogger:
    """
    Create a run logger.
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Name of experiment
        
    Returns:
        RunLogger instance
    """
    return RunLogger(log_dir, experiment_name)

