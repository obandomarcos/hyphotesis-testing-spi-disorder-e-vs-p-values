"""
Integrated Hypothesis Testing Framework for SPI with Disorder-Diffusive Models

Combines:
  1. Single-pixel imaging forward models
  2. Disorder-averaged posterior sampling (ALSS)
  3. E-value sequential hypothesis testing
  4. Optimal experimental design (OED) for measurement selection
  5. End-to-end reconstruction quality certification

References:
  - ICLR paper: Sections 3.4-3.6 on sequential hypothesis testing
  - TPAMI paper: Sections 2.1-2.4 on disorder-averaging framework
  - "Safe Testing" (Grünwald et al., 2024) for e-value theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

# Assume these modules exist and can be imported
from disorder_avg import DisorderAveragedALSS, DisorderConfig
from e_values import EValueTesting, HypothesisTestConfig
from forward_spi import SPIForwardModel, SPIConfig, PatternType, DisorderAveragedOperator, SPISequentialMeasurement


class QualityMetric(Enum):
    """Reconstruction quality metrics for hypothesis testing."""
    MSE = "mse"                        # Mean squared error
    SSIM = "ssim"                      # Structural similarity
    PSNR = "psnr"                      # Peak signal-to-noise ratio
    UNCERTAINTY = "uncertainty"        # Posterior uncertainty
    COVERAGE = "coverage"              # Credible interval coverage


class MeasurementStrategy(Enum):
    """Strategy for selecting next measurement pattern."""
    RANDOM = "random"                  # Random measurement
    ENTROPY = "entropy"                # Maximize information entropy
    INFORMATION_GAIN = "information_gain"  # Maximize expected info gain
    VARIANCE = "variance"              # Maximize posterior variance reduction
    EIGENVECTOR = "eigenvector"        # Use leading eigenvector of H^T H


@dataclass
class HypothesisTestingConfig:
    """Complete configuration for integrated hypothesis testing."""
    # SPI forward model
    spi_config: SPIConfig = field(default_factory=lambda: SPIConfig(image_dim=32, num_measurements=256))
    
    # Disorder-diffusive sampling
    alss_config: DisorderConfig = field(default_factory=DisorderConfig)
    
    # Hypothesis testing
    hypothesis_config: HypothesisTestConfig = field(default_factory=HypothesisTestConfig)
    
    # Quality certification
    quality_metric: QualityMetric = QualityMetric.MSE
    quality_threshold: float = 0.1           # δ_0: ||x̂ - x*||²₂ ≥ δ_0 (H0)
    
    # Sequential measurement design
    measurement_strategy: MeasurementStrategy = MeasurementStrategy.INFORMATION_GAIN
    max_measurements: int = 500
    min_measurements_before_decision: int = 20
    
    # Stopping criteria
    stop_on_hypothesis_reject: bool = True   # Stop when H0 rejected (good quality)
    stop_on_max_measurements: bool = True
    evalue_threshold: float = 20.0           # E-value threshold for α ≈ 0.05
    
    # Output
    return_posterior_samples: bool = True
    num_posterior_samples: int = 10
    verbose: bool = True


class IntegratedHypothesisTest:
    """
    Complete end-to-end hypothesis testing pipeline for SPI.
    
    Workflow:
      1. Generate SPI measurements sequentially
      2. Estimate effective operator (disorder-averaging)
      3. Sample posterior via disorder-diffusive ALSS
      4. Compute reconstruction quality
      5. Perform sequential e-value test
      6. Adaptively select next measurement (OED)
      7. Stop when hypothesis rejected or max measurements reached
    """
    
    def __init__(self, score_network: nn.Module, config: HypothesisTestingConfig,
                 device: str = "cpu"):
        """
        Args:
            score_network: Trained score network for prior
            config: HypothesisTestingConfig
            device: torch device
        """
        self.config = config
        self.device = device
        
        # Initialize components
        self.spi_forward = SPIForwardModel(config.spi_config, device=device)
        self.disorder_sampler = DisorderAveragedALSS(score_network, config.alss_config, device=device)
        self.evalue_tester = EValueTesting(config.hypothesis_config, device=device)
        self.disorder_estimator = DisorderAveragedOperator(config.spi_config, device=device)
        self.seq_measurement = SPISequentialMeasurement(self.spi_forward, device=device)
        
        # Testing state
        self.ground_truth = None
        self.measurement_history = []
        self.pattern_history = []
        self.reconstruction_history = []
        self.evalue_history = []
        self.quality_history = []
        self.measurement_count = 0
        self.stopping_time = None
        self.stopping_reason = None
        
        # Disorder averaging state
        self.patterns_for_disorder = []
    
    def set_ground_truth(self, x_true: torch.Tensor) -> None:
        """
        Set ground truth image (for quality assessment).
        
        Args:
            x_true: Ground truth image (num_pixels,)
        """
        self.ground_truth = x_true.to(self.device).float()
    
    def take_measurement(self, x_true: torch.Tensor,
                        adaptive: bool = False) -> float:
        """
        Take a single SPI measurement.
        
        Args:
            x_true: Image to measure
            adaptive: If True, select pattern via OED
        
        Returns:
            Scalar measurement
        """
        if adaptive and self.measurement_count > self.config.min_measurements_before_decision:
            # Adaptive measurement selection
            h = self._select_next_pattern()
        else:
            # Random measurement
            h = self.spi_forward.pattern_gen.generate_random_bernoulli(num_patterns=1)
        
        # Compute measurement
        y = self.spi_forward.forward(x_true, h).squeeze()
        
        # Record
        self.measurement_history.append(y.item())
        self.pattern_history.append(h.clone())
        self.patterns_for_disorder.append(h.clone())
        self.measurement_count += 1
        
        return y.item()
    
    def _select_next_pattern(self) -> torch.Tensor:
        """
        Select next measurement pattern via optimal experimental design.
        
        Returns:
            Optimized pattern (1, num_pixels)
        """
        strategy = self.config.measurement_strategy
        
        if strategy == MeasurementStrategy.RANDOM:
            return self.spi_forward.pattern_gen.generate_random_bernoulli(num_patterns=1)
        
        elif strategy == MeasurementStrategy.INFORMATION_GAIN:
            # Maximize I(x; y | y_{1:n-1})
            # Use posterior variance as proxy
            if len(self.reconstruction_history) > 0:
                x_recon = self.reconstruction_history[-1]
                
                def info_gain_fn(h, x):
                    pred = self.spi_forward.forward(x, h)
                    # Higher variance in prediction = higher info
                    return torch.abs(pred - 0.5)
                
                return self.seq_measurement.adaptive_next_pattern(x_recon, info_gain_fn)
            else:
                return self.spi_forward.pattern_gen.generate_random_bernoulli(num_patterns=1)
        
        elif strategy == MeasurementStrategy.ENTROPY:
            # Maximize Shannon entropy of measurement prediction
            candidates = self.spi_forward.pattern_gen.generate_random_bernoulli(num_patterns=20)
            
            if len(self.reconstruction_history) > 0:
                x_recon = self.reconstruction_history[-1]
                entropies = []
                
                for h in candidates:
                    pred = self.spi_forward.forward(x_recon, h)
                    # Entropy maximized near 0.5
                    entropy = -pred * torch.log(pred + 1e-8) - (1-pred) * torch.log(1-pred + 1e-8)
                    entropies.append(entropy.item())
                
                best_idx = np.argmax(entropies)
                return candidates[best_idx:best_idx+1]
            else:
                return candidates[0:1]
        
        else:
            return self.spi_forward.pattern_gen.generate_random_bernoulli(num_patterns=1)
    
    def reconstruct_from_measurements(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform posterior inference from current measurements.
        
        Uses disorder-averaged ALSS sampling.
        
        Returns:
            (posterior_mean, posterior_std): Reconstruction and uncertainty
        """
        if len(self.measurement_history) == 0:
            raise RuntimeError("No measurements available for reconstruction")
        
        # Get measurement matrix and vector
        H = self.spi_forward.get_measurement_matrix()
        y = torch.tensor(self.measurement_history, device=self.device, dtype=torch.float32)
        
        # Setup disorder averaging (Phase 1)
        if self.measurement_count % 50 == 0:  # Periodic update
            prior_samples = torch.randn(16, self.config.spi_config.num_pixels, device=self.device)
            self.disorder_sampler.setup_disorder_averaging(
                operators=[h.unsqueeze(0) for h in self.patterns_for_disorder[-50:]],
                prior_samples=prior_samples,
                measurement_noise_std=self.config.spi_config.measurement_noise_std
            )
        
        # Sample posterior (Phase 2)
        x_samples = self.disorder_sampler.sample(
            y.unsqueeze(0),
            num_samples=self.config.num_posterior_samples,
            verbose=False
        )  # (num_samples, 1, num_pixels)
        
        # Compute statistics
        x_samples = x_samples.squeeze(1)  # (num_samples, num_pixels)
        x_mean = x_samples.mean(dim=0)
        x_std = x_samples.std(dim=0)
        
        return x_mean, x_std
    
    def compute_reconstruction_quality(self, x_recon: torch.Tensor) -> float:
        """
        Compute reconstruction quality metric.
        
        Args:
            x_recon: Reconstructed signal
        
        Returns:
            Quality score (lower = better for MSE)
        """
        if self.ground_truth is None:
            raise RuntimeError("Ground truth not set. Use set_ground_truth() first.")
        
        metric = self.config.quality_metric
        
        if metric == QualityMetric.MSE:
            quality = F.mse_loss(x_recon, self.ground_truth).item()
        
        elif metric == QualityMetric.PSNR:
            mse = F.mse_loss(x_recon, self.ground_truth).item()
            max_val = self.ground_truth.max().item()
            quality = 10 * np.log10(max_val**2 / (mse + 1e-8))
        
        elif metric == QualityMetric.SSIM:
            # Simplified SSIM
            mean_x = x_recon.mean()
            mean_y = self.ground_truth.mean()
            var_x = x_recon.var()
            var_y = self.ground_truth.var()
            cov = ((x_recon - mean_x) * (self.ground_truth - mean_y)).mean()
            
            c1, c2 = 0.01, 0.03
            ssim = ((2*mean_x*mean_y + c1) * (2*cov + c2)) / \
                   ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))
            quality = ssim.item()
        
        else:
            quality = F.mse_loss(x_recon, self.ground_truth).item()
        
        return quality
    
    def update_hypothesis_test(self, x_recon: torch.Tensor) -> Tuple[float, bool]:
        """
        Update sequential e-value test with new reconstruction.
        
        Tests:
          H0: ∥x̂ - x*∥²₂ ≥ δ_0  (reconstruction quality is poor)
          H1: ∥x̂ - x*∥²₂ < δ_0  (reconstruction quality is good)
        
        Args:
            x_recon: Latest reconstruction
        
        Returns:
            (evalue, reject_h0): E-value and rejection decision
        """
        if self.ground_truth is None:
            raise RuntimeError("Ground truth not set for hypothesis testing")
        
        # Compute quality
        quality = self.compute_reconstruction_quality(x_recon)
        self.quality_history.append(quality)
        
        # Setup hypothesis distributions
        delta_0 = self.config.quality_threshold
        
        # H0: error distributed as N(δ_0, 1)
        # H1: error distributed as N(δ_0/2, 1)
        self.evalue_tester.config.h0_mean = torch.tensor([delta_0], device=self.device)
        self.evalue_tester.config.h1_mean = torch.tensor([delta_0 / 2], device=self.device)
        self.evalue_tester.config.h0_cov = torch.tensor([[1.0]], device=self.device)
        self.evalue_tester.config.h1_cov = torch.tensor([[1.0]], device=self.device)
        
        # Add measurement to e-value test
        y_quality = torch.tensor([quality], device=self.device)
        self.evalue_tester.add_measurement(y_quality, update_evalue=True)
        
        evalue = self.evalue_tester.cumulative_evalue
        reject = evalue >= self.config.evalue_threshold
        
        self.evalue_history.append(evalue)
        
        return evalue, reject
    
    def run_sequential_test(self, x_true: torch.Tensor,
                           adaptive_measurement: bool = True) -> Dict:
        """
        Run complete sequential hypothesis testing procedure.
        
        Performs:
          1. Take measurement (adaptive or random)
          2. Reconstruct from accumulated measurements
          3. Compute reconstruction quality
          4. Update e-value test
          5. Check stopping criteria
          6. Repeat until stopping condition met
        
        Args:
            x_true: Ground truth image
            adaptive_measurement: Whether to use OED for pattern selection
        
        Returns:
            Dictionary with testing results
        """
        self.set_ground_truth(x_true)
        
        if self.config.verbose:
            print("\n" + "="*70)
            print("Sequential Hypothesis Testing Pipeline")
            print("="*70)
            print(f"Quality threshold δ_0: {self.config.quality_threshold:.4f}")
            print(f"E-value threshold: {self.config.evalue_threshold:.1f}\n")
        
        while self.measurement_count < self.config.max_measurements:
            # Take measurement
            y = self.take_measurement(x_true, adaptive=adaptive_measurement)
            
            # Reconstruct
            if self.measurement_count >= self.config.min_measurements_before_decision:
                try:
                    x_recon, x_std = self.reconstruct_from_measurements()
                    self.reconstruction_history.append(x_recon)
                    
                    # Test hypothesis
                    evalue, reject_h0 = self.update_hypothesis_test(x_recon)
                    
                    quality = self.quality_history[-1]
                    
                    if self.config.verbose and (self.measurement_count % 10 == 0 or reject_h0):
                        print(f"N={self.measurement_count:3d}: Quality={quality:.6f}, " 
                              f"E-value={evalue:.2f}, Reject H0? {reject_h0}")
                    
                    # Check stopping criteria
                    if self.config.stop_on_hypothesis_reject and reject_h0:
                        self.stopping_time = self.measurement_count
                        self.stopping_reason = "H0 rejected (good quality)"
                        break
                
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Reconstruction failed at N={self.measurement_count}: {str(e)}")
                    continue
            
            # Check max measurements
            if self.config.stop_on_max_measurements and self.measurement_count >= self.config.max_measurements:
                self.stopping_time = self.measurement_count
                self.stopping_reason = "Max measurements reached"
                break
        
        # Compile results
        results = {
            'stopping_time': self.stopping_time,
            'stopping_reason': self.stopping_reason,
            'total_measurements': self.measurement_count,
            'final_evalue': self.evalue_history[-1] if self.evalue_history else 1.0,
            'final_quality': self.quality_history[-1] if self.quality_history else None,
            'quality_threshold': self.config.quality_threshold,
            'quality_history': self.quality_history,
            'evalue_history': self.evalue_history,
            'measurement_count_sequence': list(range(1, len(self.quality_history) + 1)),
        }
        
        if self.config.return_posterior_samples and len(self.reconstruction_history) > 0:
            results['final_reconstruction'] = self.reconstruction_history[-1]
        
        if self.config.verbose:
            print("\n" + "="*70)
            print("Testing Complete")
            print("="*70)
            print(f"Stopping time: {self.stopping_time} measurements")
            print(f"Stopping reason: {self.stopping_reason}")
            print(f"Final quality: {results['final_quality']:.6f}")
            print(f"Final e-value: {results['final_evalue']:.2f}")
            print(f"H0 rejected? {results['final_evalue'] >= self.config.evalue_threshold}")
            print("="*70)
        
        return results


class HypothesisTestingReport:
    """Generate analysis report from testing results."""
    
    @staticmethod
    def plot_convergence(results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot quality and e-value convergence.
        
        Args:
            results: Results dictionary from run_sequential_test()
            save_path: Path to save figure (if None, displays)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Quality convergence
        ax = axes[0]
        ax.plot(results['measurement_count_sequence'], results['quality_history'], 'b-o', linewidth=2)
        ax.axhline(results['quality_threshold'], color='r', linestyle='--', label='Quality threshold')
        ax.set_xlabel('Number of Measurements')
        ax.set_ylabel('Quality (MSE)')
        ax.set_title('Reconstruction Quality Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # E-value convergence
        ax = axes[1]
        ax.semilogy(results['measurement_count_sequence'], results['evalue_history'], 'g-s', linewidth=2)
        ax.axhline(20.0, color='r', linestyle='--', label='Decision boundary (α≈0.05)')
        ax.set_xlabel('Number of Measurements')
        ax.set_ylabel('E-value (log scale)')
        ax.set_title('E-value Accumulation (Anytime-Valid)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def summary_stats(results: Dict) -> Dict:
        """
        Compute summary statistics.
        
        Args:
            results: Results dictionary
        
        Returns:
            Dictionary with summary statistics
        """
        quality_hist = results['quality_history']
        evalue_hist = results['evalue_history']
        
        return {
            'stopping_time': results['stopping_time'],
            'measurements_efficiency': results['stopping_time'] / results['total_measurements'] if results['total_measurements'] > 0 else 0,
            'initial_quality': quality_hist[0] if quality_hist else None,
            'final_quality': quality_hist[-1] if quality_hist else None,
            'quality_improvement': (quality_hist[0] - quality_hist[-1]) if quality_hist and len(quality_hist) > 1 else None,
            'quality_convergence_rate': np.mean(np.diff(quality_hist)) if quality_hist and len(quality_hist) > 1 else None,
            'final_evalue': evalue_hist[-1] if evalue_hist else None,
            'evalue_growth_rate': np.mean(np.diff(np.log(evalue_hist + 1e-8))) if evalue_hist and len(evalue_hist) > 1 else None,
        }


def main_example():
    """Example of complete integrated hypothesis testing."""
    
    print("=" * 70)
    print("Integrated Hypothesis Testing for SPI with Disorder-Diffusive Models")
    print("=" * 70)
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Setup configuration
    config = HypothesisTestingConfig(
        spi_config=SPIConfig(
            image_dim=16,
            num_measurements=200,
            pattern_type=PatternType.RANDOM_BERNOULLI,
            measurement_noise_std=0.05
        ),
        alss_config=DisorderConfig(
            num_steps=50,
            levy_alpha=1.5,
            likelihood_weight=1.0
        ),
        quality_threshold=0.1,
        measurement_strategy=MeasurementStrategy.INFORMATION_GAIN,
        max_measurements=100,
        evalue_threshold=20.0,
        verbose=True
    )
    
    # Create dummy score network
    class SimpleScoreNet(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 64),
                nn.ReLU(),
                nn.Linear(64, dim)
            )
        
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = t.unsqueeze(-1).float() / 1000.0
            x_and_t = torch.cat([x, t_emb], dim=-1)
            return self.net(x_and_t)
    
    score_net = SimpleScoreNet(config.spi_config.num_pixels).to(device)
    
    # Create testing pipeline
    tester = IntegratedHypothesisTest(score_net, config, device=device)
    
    # Create synthetic test image
    x_true = torch.rand(config.spi_config.num_pixels, device=device) * 0.8 + 0.1
    
    # Run test
    results = tester.run_sequential_test(x_true, adaptive_measurement=True)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 70)
    stats = HypothesisTestingReport.summary_stats(results)
    for key, value in stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.6f}")
            else:
                print(f"  {key:30s}: {value}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
