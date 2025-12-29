"""
Metrics, Logging, and Evaluation Framework

Implements:
  1. Comprehensive evaluation metrics (PSNR, SSIM, MSE, etc.)
  2. Uncertainty quantification and coverage analysis
  3. Sequential convergence tracking
  4. Structured logging and experiment management
  5. Visualization utilities for results analysis

References:
  - Standard image quality metrics (PSNR, SSIM, MS-SSIM)
  - Uncertainty quantification for posterior sampling
  - Experimental logging best practices
  - ICLR/TPAMI paper metrics and ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import csv
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import os


class MetricType(Enum):
    """Types of metrics for evaluation."""
    RECONSTRUCTION = "reconstruction"
    UNCERTAINTY = "uncertainty"
    HYPOTHESIS = "hypothesis"
    CONVERGENCE = "convergence"
    EFFICIENCY = "efficiency"


@dataclass
class ReconstructionMetrics:
    """Reconstruction quality metrics."""
    mse: float = 0.0                   # Mean squared error
    rmse: float = 0.0                  # Root mean squared error
    mae: float = 0.0                   # Mean absolute error
    psnr: float = 0.0                  # Peak signal-to-noise ratio (dB)
    ssim: float = 0.0                  # Structural similarity index
    ms_ssim: float = 0.0               # Multi-scale SSIM
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics."""
    mean_std: float = 0.0              # Mean posterior standard deviation
    std_std: float = 0.0               # Std of posterior std
    credible_interval_coverage: float = 0.0  # Fraction of ground truth in CI
    credible_interval_width: float = 0.0    # Average CI width
    calibration_error: float = 0.0     # Expected calibration error
    negative_log_likelihood: float = 0.0    # NLL of posterior samples
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HypothesisMetrics:
    """Hypothesis testing metrics."""
    final_evalue: float = 1.0          # Final e-value
    stopping_time: Optional[int] = None    # When test stopped
    h0_rejected: bool = False          # Whether H0 was rejected
    anytime_valid: bool = True         # Whether test is anytime-valid
    power_estimate: float = 0.0        # Estimated power
    type_i_error: float = 0.0          # Type I error rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConvergenceMetrics:
    """Sequential convergence tracking."""
    measurement_numbers: List[int] = field(default_factory=list)
    quality_sequence: List[float] = field(default_factory=list)
    evalue_sequence: List[float] = field(default_factory=list)
    uncertainty_sequence: List[float] = field(default_factory=list)
    
    # Convergence rates
    quality_convergence_rate: float = 0.0   # dQuality/dN
    evalue_growth_rate: float = 0.0         # d(log E)/dN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        return d


@dataclass
class EfficiencyMetrics:
    """Measurement efficiency metrics."""
    total_measurements: int = 0
    effective_measurements: int = 0    # After removing redundant
    adaptive_vs_random_ratio: float = 1.0
    information_per_measurement: float = 0.0
    stopping_time_ratio: float = 0.0  # stopping_time / max_measurements
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


class ImageQualityMetrics:
    """Compute standard image quality metrics."""
    
    @staticmethod
    def mse(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
        """Mean squared error."""
        return F.mse_loss(x_pred, x_true).item()
    
    @staticmethod
    def rmse(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
        """Root mean squared error."""
        mse = F.mse_loss(x_pred, x_true).item()
        return np.sqrt(mse)
    
    @staticmethod
    def mae(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
        """Mean absolute error."""
        return F.l1_loss(x_pred, x_true).item()
    
    @staticmethod
    def psnr(x_pred: torch.Tensor, x_true: torch.Tensor, 
             data_range: float = 1.0) -> float:
        """
        Peak signal-to-noise ratio (dB).
        
        PSNR = 20 * log10(max_val / sqrt(MSE))
        
        Args:
            x_pred: Predicted signal
            x_true: Ground truth signal
            data_range: Maximum value of signal
        
        Returns:
            PSNR in dB
        """
        mse = F.mse_loss(x_pred, x_true).item()
        if mse < 1e-8:
            return 100.0  # Perfect reconstruction
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        return float(psnr)
    
    @staticmethod
    def ssim(x_pred: torch.Tensor, x_true: torch.Tensor,
             kernel_size: int = 11, sigma: float = 1.5,
             data_range: float = 1.0) -> float:
        """
        Structural similarity index (SSIM).
        
        SSIM = (2μ_x μ_y + c1)(2σ_xy + c2) / ((μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2))
        
        Args:
            x_pred: Predicted signal
            x_true: Ground truth signal
            kernel_size: Gaussian kernel size
            sigma: Gaussian kernel standard deviation
            data_range: Maximum value of signal
        
        Returns:
            SSIM ∈ [-1, 1]
        """
        # Constants for numerical stability
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        
        # Compute Gaussian kernel
        kernel = ImageQualityMetrics._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x_pred.device).unsqueeze(0).unsqueeze(0)
        
        # Ensure 4D for conv2d: (batch, channel, height, width)
        if x_pred.dim() == 1:
            x_pred = x_pred.view(1, 1, -1, 1)
            x_true = x_true.view(1, 1, -1, 1)
            pad = (kernel_size - 1) // 2
            padding = (pad, pad, 0, 0)
        else:
            padding = (kernel_size - 1) // 2
        
        # Pad for same-size output
        x_pred_padded = F.pad(x_pred, (padding, padding, padding, padding) if x_pred.dim() == 4 else (padding, padding))
        x_true_padded = F.pad(x_true, (padding, padding, padding, padding) if x_true.dim() == 4 else (padding, padding))
        
        # Compute local means
        mu_x = F.conv2d(x_pred_padded, kernel, padding=0)
        mu_y = F.conv2d(x_true_padded, kernel, padding=0)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Compute local variances
        sigma_x_sq = F.conv2d(x_pred_padded**2, kernel, padding=0) - mu_x_sq
        sigma_y_sq = F.conv2d(x_true_padded**2, kernel, padding=0) - mu_y_sq
        sigma_xy = F.conv2d(x_pred_padded * x_true_padded, kernel, padding=0) - mu_xy
        
        # Compute SSIM map
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        return float(ssim_map.mean().item())
    
    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = gauss.unsqueeze(-1) @ gauss.unsqueeze(-1)
        kernel = kernel / kernel.sum()
        return kernel
    
    @staticmethod
    def ms_ssim(x_pred: torch.Tensor, x_true: torch.Tensor,
                scales: int = 5, data_range: float = 1.0) -> float:
        """
        Multi-scale SSIM.
        
        Computes SSIM at multiple scales and averages.
        
        Args:
            x_pred: Predicted signal
            x_true: Ground truth signal
            scales: Number of scales
            data_range: Maximum value of signal
        
        Returns:
            MS-SSIM ∈ [-1, 1]
        """
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                               device=x_pred.device)[:scales]
        weights = weights / weights.sum()
        
        mssim = 0.0
        x_pred_scale = x_pred
        x_true_scale = x_true
        
        for i in range(scales):
            ssim_val = ImageQualityMetrics.ssim(x_pred_scale, x_true_scale, 
                                               data_range=data_range)
            mssim += weights[i].item() * ssim_val
            
            if i < scales - 1:
                # Downsample for next scale
                if x_pred_scale.dim() == 1:
                    x_pred_scale = x_pred_scale[::2]
                    x_true_scale = x_true_scale[::2]
                else:
                    x_pred_scale = F.avg_pool2d(x_pred_scale, kernel_size=2, stride=2)
                    x_true_scale = F.avg_pool2d(x_true_scale, kernel_size=2, stride=2)
        
        return float(mssim)


class UncertaintyQuantification:
    """Compute uncertainty quantification metrics."""
    
    @staticmethod
    def credible_interval_coverage(x_mean: torch.Tensor, x_std: torch.Tensor,
                                   x_true: torch.Tensor,
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute coverage of credible interval.
        
        Assumes posterior is Gaussian: CI = [μ - z*σ, μ + z*σ]
        
        Args:
            x_mean: Posterior mean
            x_std: Posterior standard deviation
            x_true: Ground truth
            confidence: Confidence level (e.g., 0.95)
        
        Returns:
            (coverage, avg_width): Coverage probability and average interval width
        """
        z_score = torch.tensor(2.0, device=x_mean.device)  # ~95% for Gaussian
        
        lower = x_mean - z_score * x_std
        upper = x_mean + z_score * x_std
        
        # Check if ground truth is in CI
        in_ci = (x_true >= lower) & (x_true <= upper)
        coverage = in_ci.float().mean().item()
        
        width = (upper - lower).mean().item()
        
        return coverage, width
    
    @staticmethod
    def calibration_error(samples: torch.Tensor, x_true: torch.Tensor,
                         num_bins: int = 10) -> float:
        """
        Expected calibration error (ECE).
        
        Measures alignment between predicted confidence and actual accuracy.
        
        Args:
            samples: Posterior samples (num_samples, dim)
            x_true: Ground truth
            num_bins: Number of confidence bins
        
        Returns:
            ECE ∈ [0, 1]
        """
        # Compute posterior variance per dimension
        posterior_var = samples.var(dim=0)
        posterior_std = torch.sqrt(posterior_var)
        
        # Errors
        errors = torch.abs(samples - x_true.unsqueeze(0))
        
        # Bin by confidence (inverse std)
        ece = 0.0
        for i in range(num_bins):
            bin_threshold = i / num_bins
            mask = posterior_std >= torch.quantile(posterior_std, bin_threshold)
            
            if mask.sum() > 0:
                actual_error = errors[:, mask].mean().item()
                expected_error = (1 - posterior_std[mask]).mean().item()
                ece += np.abs(actual_error - expected_error) / num_bins
        
        return float(ece)
    
    @staticmethod
    def negative_log_likelihood(samples: torch.Tensor, x_true: torch.Tensor,
                               x_mean: torch.Tensor, x_cov: torch.Tensor) -> float:
        """
        Negative log-likelihood under fitted Gaussian.
        
        NLL = -log p(x_true | μ, Σ)
        
        Args:
            samples: Posterior samples (for covariance computation)
            x_true: Ground truth
            x_mean: Posterior mean
            x_cov: Posterior covariance
        
        Returns:
            NLL (higher is worse)
        """
        try:
            # Log determinant of covariance
            log_det = torch.logdet(x_cov).item()
            
            # Mahalanobis distance
            residual = x_true - x_mean
            cov_inv = torch.linalg.inv(x_cov)
            mahal = (residual @ cov_inv @ residual.T).item()
            
            # NLL
            dim = x_true.shape[0]
            nll = 0.5 * (log_det + mahal + dim * np.log(2 * np.pi))
        except:
            # Fallback for singular covariance
            nll = torch.norm(x_true - x_mean).item()
        
        return float(nll)


class ExperimentLogger:
    """Structured experiment logging and tracking."""
    
    def __init__(self, experiment_name: str, output_dir: str = "./logs"):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of experiment
            output_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(output_dir, f"{experiment_name}_{self.timestamp}")
        
        # Create directory
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.exp_dir, "experiment.log"))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Metrics storage
        self.metrics_history = {
            'reconstruction': [],
            'uncertainty': [],
            'hypothesis': [],
            'convergence': [],
            'efficiency': []
        }
        
        # CSV writers
        self.csv_files = {}
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_config(self, config: Any) -> None:
        """Log configuration."""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w') as f:
            if hasattr(config, 'asdict'):
                json.dump(config.asdict(), f, indent=2, default=str)
            elif isinstance(config, dict):
                json.dump(config, f, indent=2, default=str)
            else:
                json.dump(str(config), f, indent=2)
        self.log_info(f"Config saved to {config_path}")
    
    def log_metrics(self, metric_type: MetricType, metrics: Any) -> None:
        """
        Log metrics of specific type.
        
        Args:
            metric_type: Type of metrics
            metrics: Metrics object or dictionary
        """
        if isinstance(metrics, dict):
            metrics_dict = metrics
        else:
            metrics_dict = metrics.to_dict()
        
        self.metrics_history[metric_type.value].append(metrics_dict)
        
        # Also write to CSV
        self._write_to_csv(metric_type, metrics_dict)
        
        self.log_info(f"{metric_type.value}: {metrics_dict}")
    
    def _write_to_csv(self, metric_type: MetricType, metrics_dict: Dict) -> None:
        """Write metrics to CSV file."""
        csv_path = os.path.join(self.exp_dir, f"{metric_type.value}_metrics.csv")
        
        if metric_type not in self.csv_files:
            self.csv_files[metric_type] = open(csv_path, 'w', newline='')
            writer = csv.DictWriter(self.csv_files[metric_type], fieldnames=metrics_dict.keys())
            writer.writeheader()
            self.csv_files[metric_type]._writer = writer
        
        self.csv_files[metric_type]._writer.writerow(metrics_dict)
        self.csv_files[metric_type].flush()
    
    def save_results(self, results: Dict) -> None:
        """
        Save complete results to JSON.
        
        Args:
            results: Results dictionary
        """
        results_path = os.path.join(self.exp_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.log_info(f"Results saved to {results_path}")
    
    def close(self) -> None:
        """Close all file handles."""
        for f in self.csv_files.values():
            f.close()
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all logged metrics."""
        summary = {}
        for metric_type, metrics_list in self.metrics_history.items():
            if metrics_list:
                # Average numeric metrics
                summary[metric_type] = {}
                first_metric = metrics_list[0]
                for key in first_metric.keys():
                    values = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
                    if values:
                        summary[metric_type][key] = {
                            'final': values[-1],
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
        return summary


class MetricsAnalyzer:
    """Analyze and compare metrics across experiments."""
    
    @staticmethod
    def compare_experiments(experiment_dirs: List[str]) -> Dict:
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_dirs: List of experiment directories
        
        Returns:
            Comparison dictionary
        """
        comparison = {}
        
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            results_file = os.path.join(exp_dir, "results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                comparison[exp_name] = results
        
        return comparison
    
    @staticmethod
    def compute_efficiency_ratio(adaptive_results: Dict,
                                random_results: Dict) -> Dict:
        """
        Compare efficiency of adaptive vs. random measurement.
        
        Args:
            adaptive_results: Results with adaptive measurement
            random_results: Results with random measurement
        
        Returns:
            Efficiency metrics
        """
        return {
            'stopping_time_reduction': (
                (random_results.get('stopping_time', 1) - 
                 adaptive_results.get('stopping_time', 1)) / 
                random_results.get('stopping_time', 1)
            ),
            'quality_improvement': (
                (random_results.get('final_quality', 1) - 
                 adaptive_results.get('final_quality', 1)) / 
                random_results.get('final_quality', 1)
            ),
        }


def main_example():
    """Example demonstrating metrics and logging."""
    
    print("=" * 70)
    print("Metrics, Logging, and Evaluation Framework")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create logger
    logger = ExperimentLogger("test_experiment", output_dir="./test_logs")
    logger.log_info("Starting metrics demonstration")
    
    # Create synthetic data
    x_true = torch.rand(100, device=device)
    x_pred = x_true + 0.05 * torch.randn_like(x_true)
    x_std = 0.1 * torch.ones_like(x_true)
    
    print("\n1. Image Quality Metrics")
    print("-" * 70)
    
    mse = ImageQualityMetrics.mse(x_pred, x_true)
    rmse = ImageQualityMetrics.rmse(x_pred, x_true)
    mae = ImageQualityMetrics.mae(x_pred, x_true)
    psnr = ImageQualityMetrics.psnr(x_pred, x_true)
    ssim = ImageQualityMetrics.ssim(x_pred.unsqueeze(0).unsqueeze(0),
                                     x_true.unsqueeze(0).unsqueeze(0))
    
    recon_metrics = ReconstructionMetrics(
        mse=mse, rmse=rmse, mae=mae, psnr=psnr, ssim=ssim, ms_ssim=ssim
    )
    
    print(f"  MSE:  {recon_metrics.mse:.6f}")
    print(f"  RMSE: {recon_metrics.rmse:.6f}")
    print(f"  MAE:  {recon_metrics.mae:.6f}")
    print(f"  PSNR: {recon_metrics.psnr:.2f} dB")
    print(f"  SSIM: {recon_metrics.ssim:.4f}")
    
    logger.log_metrics(MetricType.RECONSTRUCTION, recon_metrics)
    
    print("\n2. Uncertainty Quantification")
    print("-" * 70)
    
    coverage, ci_width = UncertaintyQuantification.credible_interval_coverage(
        x_pred, x_std, x_true
    )
    
    unc_metrics = UncertaintyMetrics(
        mean_std=x_std.mean().item(),
        std_std=x_std.std().item(),
        credible_interval_coverage=coverage,
        credible_interval_width=ci_width
    )
    
    print(f"  Mean posterior std:      {unc_metrics.mean_std:.6f}")
    print(f"  CI coverage:             {unc_metrics.credible_interval_coverage:.4f}")
    print(f"  Average CI width:        {unc_metrics.credible_interval_width:.6f}")
    
    logger.log_metrics(MetricType.UNCERTAINTY, unc_metrics)
    
    print("\n3. Hypothesis Testing Metrics")
    print("-" * 70)
    
    hyp_metrics = HypothesisMetrics(
        final_evalue=25.5,
        stopping_time=42,
        h0_rejected=True,
        power_estimate=0.92
    )
    
    print(f"  Final e-value:     {hyp_metrics.final_evalue:.2f}")
    print(f"  Stopping time:     {hyp_metrics.stopping_time}")
    print(f"  H0 rejected:       {hyp_metrics.h0_rejected}")
    print(f"  Power estimate:    {hyp_metrics.power_estimate:.4f}")
    
    logger.log_metrics(MetricType.HYPOTHESIS, hyp_metrics)
    
    print("\n4. Convergence Tracking")
    print("-" * 70)
    
    conv_metrics = ConvergenceMetrics(
        measurement_numbers=list(range(1, 51)),
        quality_sequence=[0.5 / (1 + 0.05*n) for n in range(1, 51)],
        evalue_sequence=[1.0 * (1.2 ** n) for n in range(1, 51)]
    )
    
    print(f"  Total measurements:      {len(conv_metrics.measurement_numbers)}")
    print(f"  Quality range:           [{min(conv_metrics.quality_sequence):.4f}, {max(conv_metrics.quality_sequence):.4f}]")
    print(f"  E-value range (log):     [{np.log(min(conv_metrics.evalue_sequence)):.2f}, {np.log(max(conv_metrics.evalue_sequence)):.2f}]")
    
    logger.log_metrics(MetricType.CONVERGENCE, conv_metrics)
    
    print("\n5. Efficiency Metrics")
    print("-" * 70)
    
    eff_metrics = EfficiencyMetrics(
        total_measurements=100,
        effective_measurements=42,
        adaptive_vs_random_ratio=0.84,
        stopping_time_ratio=0.42
    )
    
    print(f"  Total measurements:      {eff_metrics.total_measurements}")
    print(f"  Effective measurements:  {eff_metrics.effective_measurements}")
    print(f"  Adaptive efficiency:     {eff_metrics.adaptive_vs_random_ratio:.2%}")
    print(f"  Stopping time ratio:     {eff_metrics.stopping_time_ratio:.2%}")
    
    logger.log_metrics(MetricType.EFFICIENCY, eff_metrics)
    
    # Save summary
    summary = logger.get_metrics_summary()
    logger.save_results({'summary': summary})
    logger.close()
    
    print("\n" + "=" * 70)
    print(f"Logs saved to {logger.exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
