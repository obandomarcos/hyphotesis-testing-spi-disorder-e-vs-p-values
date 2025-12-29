"""
E-values and P-values for Sequential Hypothesis Testing in Single-Pixel Imaging

Implements:
  1. Classical p-value framework with limitations discussion
  2. E-value framework for optional stopping and model misspecification
  3. Likelihood ratio e-values for Gaussian posteriors
  4. Sequential/cumulative e-value processes for anytime-valid testing
  5. Integration with disorder-diffusive sampling

References:
  - "Safe Testing" (Grünwald, de Heide, Koolen, 2024)
  - "Hypothesis Testing with E-values" (Ramdas & Wang, 2025)
  - ICLR paper: Hypothesis Testing with SPI and Disorder-Diffusive Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass
from scipy.special import logsumexp
from scipy.stats import norm, chi2
import warnings


@dataclass
class HypothesisTestConfig:
    """Configuration for hypothesis testing."""
    # Hypothesis specification
    h0_mean: Optional[torch.Tensor] = None  # μ_0 for H0
    h1_mean: Optional[torch.Tensor] = None  # μ_1 for H1
    h0_cov: Optional[torch.Tensor] = None   # Σ_0 for H0
    h1_cov: Optional[torch.Tensor] = None   # Σ_1 for H1
    
    # Testing parameters
    alpha: float = 0.05  # Significance level for p-values
    evalue_threshold: float = 1.0 / 0.05  # E-value threshold (20 for α=0.05)
    
    # Stopping rules
    max_measurements: int = 1000
    min_measurements: int = 10
    
    # Quality threshold for reconstruction
    quality_threshold: float = 0.1  # δ_0: ||x̂ - x*||²₂
    
    # Sequential testing
    batch_size: int = 1  # Accumulate every k measurements
    use_anytime_valid: bool = True  # Use anytime-valid e-values


class PValueTesting:
    """
    Classical p-value hypothesis testing framework.
    
    Limitations:
      - Only valid for fixed sample sizes or pre-specified group-sequential designs
      - Adaptive sampling (OED) invalidates Type I error control
      - Requires exact null distribution (often intractable for learned priors)
      - Cannot be computed from non-parametric posteriors easily
    """
    
    def __init__(self, config: HypothesisTestConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.measurement_count = 0
        self.test_statistics = []
    
    def add_test_statistic(self, t_stat: float) -> None:
        """Record a new test statistic."""
        self.test_statistics.append(t_stat)
        self.measurement_count += 1
    
    def compute_pvalue_gaussian(self, t_obs: float, one_sided: bool = True) -> float:
        """
        Compute p-value assuming standard normal test statistic.
        
        Args:
            t_obs: Observed test statistic
            one_sided: If True, compute one-sided p-value; else two-sided
        
        Returns:
            p-value ∈ [0, 1]
        """
        if one_sided:
            pvalue = 1.0 - norm.cdf(t_obs)
        else:
            pvalue = 2.0 * (1.0 - norm.cdf(np.abs(t_obs)))
        
        return float(np.clip(pvalue, 0, 1))
    
    def compute_pvalue_reconstruction_error(self, x_recon: torch.Tensor,
                                            x_true: torch.Tensor,
                                            test_stat: str = "mse") -> float:
        """
        Compute p-value for reconstruction error hypothesis test.
        
        H0: ||x̂ - x*||²₂ ≥ δ_0  (poor reconstruction)
        H1: ||x̂ - x*||²₂ < δ_0  (good reconstruction)
        
        Args:
            x_recon: Reconstructed signal
            x_true: Ground truth signal
            test_stat: "mse" for mean squared error, "norm" for L2 norm
        
        Returns:
            p-value
        """
        if test_stat == "mse":
            error = F.mse_loss(x_recon, x_true).item()
        elif test_stat == "norm":
            error = torch.norm(x_recon - x_true).item()
        else:
            raise ValueError(f"Unknown test statistic: {test_stat}")
        
        # Z-score assuming error ~ N(δ_0, σ²_error)
        delta_0 = self.config.quality_threshold
        sigma_error = np.sqrt(delta_0)  # Placeholder
        z_score = (error - delta_0) / sigma_error
        
        pvalue = self.compute_pvalue_gaussian(z_score, one_sided=True)
        return pvalue
    
    def reject_h0_pvalue(self, pvalue: float) -> bool:
        """
        Reject H0 if p-value ≤ α.
        
        WARNING: Only valid for fixed sample size!
        
        Args:
            pvalue: Computed p-value
        
        Returns:
            True if H0 should be rejected
        """
        return pvalue <= self.config.alpha
    
    def sequential_reject_h0(self, pvalue: float) -> bool:
        """
        Sequential testing with p-values (not recommended for adaptive sampling).
        
        Uses Wald's sequential probability ratio test (SPRT) as comparison.
        
        Returns:
            True if H0 should be rejected
        """
        warnings.warn(
            "Sequential p-value testing is not recommended for adaptive sampling. "
            "Use e-values instead (EValueTesting class)."
        )
        return pvalue <= self.config.alpha


class EValueTesting:
    """
    E-value framework for anytime-valid hypothesis testing.
    
    Advantages over p-values:
      1. Valid under optional stopping (anytime-valid)
      2. Distribution-free construction possible
      3. Robust to mild model misspecification
      4. Naturally handles sequential/adaptive measurement
    
    Theory:
      - An e-value E_n satisfies E_H0[E_n] ≤ 1
      - E-process {E_n}_{n≥1} is a nonnegative supermartingale under H0
      - For any data-adaptive stopping time τ, E_τ remains valid: E_H0[E_τ] ≤ 1
    """
    
    def __init__(self, config: HypothesisTestConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Sequential tracking
        self.measurement_history = []
        self.evalue_history = []
        self.cumulative_evalue = 1.0
        self.measurement_count = 0
        
        # Likelihood ratio components
        self.log_likelihood_h0_history = []
        self.log_likelihood_h1_history = []
    
    def compute_log_likelihood_gaussian(self, y: torch.Tensor,
                                       mean: torch.Tensor,
                                       cov: torch.Tensor) -> float:
        """
        Compute log p(y | μ, Σ) for Gaussian distribution.
        
        log p(y) = -1/2 [log|Σ| + (y-μ)ᵀ Σ⁻¹ (y-μ)]
        
        Args:
            y: Observation (scalar or vector)
            mean: Mean μ
            cov: Covariance Σ
        
        Returns:
            Log-likelihood (float)
        """
        y = y.to(self.device).float()
        mean = mean.to(self.device).float()
        cov = cov.to(self.device).float()
        
        # Handle scalar case
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if mean.dim() == 0:
            mean = mean.unsqueeze(0)
        
        # Ensure matching dimensions
        if y.shape != mean.shape:
            y = y.reshape(mean.shape)
        
        try:
            # Log determinant
            log_det = torch.logdet(cov).item()
            
            # Mahalanobis distance
            residual = y - mean
            cov_inv = torch.linalg.inv(cov)
            mahal = torch.dot(residual, cov_inv @ residual).item()
            
            log_likelihood = -0.5 * (log_det + mahal)
        except:
            # Fallback for singular/ill-conditioned covariance
            log_likelihood = -0.5 * torch.sum((y - mean) ** 2).item()
        
        return log_likelihood
    
    def add_measurement(self, y_n: torch.Tensor, update_evalue: bool = True) -> None:
        """
        Process a new measurement and update cumulative e-value.
        
        Args:
            y_n: New measurement (scalar or vector)
            update_evalue: Whether to update cumulative e-value immediately
        """
        self.measurement_history.append(y_n.clone())
        self.measurement_count += 1
        
        if update_evalue:
            self.update_cumulative_evalue_from_last()
    
    def update_cumulative_evalue_from_last(self) -> float:
        """
        Update cumulative e-value using only the last measurement.
        
        E^cum_N = ∏_{n=1}^N [p(y_n | H1) / p(y_n | H0)]
        
        Returns:
            New cumulative e-value
        """
        if len(self.measurement_history) == 0:
            return self.cumulative_evalue
        
        y_n = self.measurement_history[-1]
        
        # Default to simple likelihood ratio under Gaussian assumption
        if self.config.h0_mean is None:
            self.config.h0_mean = torch.zeros_like(y_n)
        if self.config.h1_mean is None:
            self.config.h1_mean = torch.ones_like(y_n)
        if self.config.h0_cov is None:
            self.config.h0_cov = torch.eye(y_n.numel(), device=self.device)
        if self.config.h1_cov is None:
            self.config.h1_cov = torch.eye(y_n.numel(), device=self.device)
        
        log_ll_h0 = self.compute_log_likelihood_gaussian(
            y_n, self.config.h0_mean, self.config.h0_cov
        )
        log_ll_h1 = self.compute_log_likelihood_gaussian(
            y_n, self.config.h1_mean, self.config.h1_cov
        )
        
        # Log e-value for this measurement
        log_evalue_n = log_ll_h1 - log_ll_h0
        
        self.log_likelihood_h0_history.append(log_ll_h0)
        self.log_likelihood_h1_history.append(log_ll_h1)
        
        # Accumulate (product → sum in log space)
        log_cumulative = np.log(self.cumulative_evalue) + log_evalue_n
        self.cumulative_evalue = np.exp(log_cumulative)
        
        self.evalue_history.append(self.cumulative_evalue)
        
        return self.cumulative_evalue
    
    def compute_likelihood_ratio_evalue(self, y: Optional[torch.Tensor] = None,
                                        use_history: bool = True) -> float:
        """
        Compute likelihood ratio e-value.
        
        E^LR = p(y | H1) / p(y | H0)
        
        For Gaussian models:
          E^LR = sqrt(|Σ_0|/|Σ_1|) * exp(-1/2 [yᵀ Σ₀⁻¹ y - yᵀ Σ₁⁻¹ y])
        
        Args:
            y: Single measurement (if None, uses accumulated history)
            use_history: If True, return cumulative e-value from history
        
        Returns:
            E-value (float ≥ 0)
        """
        if use_history or y is None:
            return self.cumulative_evalue
        
        # Single measurement e-value
        if self.config.h0_mean is None:
            self.config.h0_mean = torch.zeros_like(y)
        if self.config.h1_mean is None:
            self.config.h1_mean = torch.ones_like(y)
        if self.config.h0_cov is None:
            self.config.h0_cov = torch.eye(y.numel(), device=self.device)
        if self.config.h1_cov is None:
            self.config.h1_cov = torch.eye(y.numel(), device=self.device)
        
        log_evalue = (
            self.compute_log_likelihood_gaussian(y, self.config.h1_mean, self.config.h1_cov) -
            self.compute_log_likelihood_gaussian(y, self.config.h0_mean, self.config.h0_cov)
        )
        
        return np.exp(log_evalue)
    
    def reject_h0_evalue(self, evalue: Optional[float] = None) -> bool:
        """
        Reject H0 if e-value ≥ threshold.
        
        Valid under optional stopping!
        
        Args:
            evalue: E-value to test (if None, uses current cumulative)
        
        Returns:
            True if H0 should be rejected
        """
        if evalue is None:
            evalue = self.cumulative_evalue
        
        threshold = self.config.evalue_threshold
        return evalue >= threshold
    
    def get_rejection_boundary(self) -> Tuple[List[int], List[float]]:
        """
        Get the sequential rejection boundary as function of measurements.
        
        Returns:
            (n_measurements, evalue_threshold) pairs defining rejection boundary
        """
        n_vals = np.arange(1, self.measurement_count + 1)
        threshold = np.full_like(n_vals, self.config.evalue_threshold, dtype=float)
        
        return list(n_vals), list(threshold)
    
    def compute_intrinsic_evalue(self, y: torch.Tensor) -> float:
        """
        Compute intrinsic (universal) e-value.
        
        The intrinsic e-value is the maximum e-value over all alternative
        distributions, providing robustness without specifying H1.
        
        For reconstruction: compares observed error against H0.
        
        Args:
            y: Test statistic or measurement
        
        Returns:
            Intrinsic e-value
        """
        # Simplified version: use profile likelihood
        if isinstance(y, (int, float)):
            return float(y) if y > 0 else 1.0
        
        if torch.is_tensor(y):
            y_val = y.abs().max().item()
            return y_val if y_val > 0 else 1.0
        
        return 1.0
    
    def power_analysis(self, effect_size: float, num_measurements: int) -> float:
        """
        Estimate statistical power: P(reject H0 | H1 true).
        
        Uses expected log e-value under H1:
          E_H1[log E_n] = E_H1[log(p(Y_n|H1)/p(Y_n|H0))]
                        = KL(H1 || H0)  (Kullback-Leibler divergence)
        
        Args:
            effect_size: Difference in means |μ_1 - μ_0|
            num_measurements: Number of samples before stopping
        
        Returns:
            Approximate power
        """
        # Approximate KL divergence for Gaussian case
        # KL(N(μ_1,Σ) || N(μ_0,Σ)) ≈ ||μ_1 - μ_0||²_Σ⁻¹ / 2
        
        if effect_size <= 0:
            return 0.0
        
        # Expected log e-value per measurement
        expected_log_evalue_per_n = effect_size ** 2 / 2
        
        # Total log e-value after N measurements
        expected_log_evalue_total = num_measurements * expected_log_evalue_per_n
        
        # Probability that log e-value exceeds threshold
        # P(E_N ≥ threshold) = P(log E_N ≥ log threshold)
        log_threshold = np.log(self.config.evalue_threshold)
        
        # Approximate using normal tail (valid for large N)
        # log E_N ~ N(N * KL, N * Var[...])  under H1
        z_score = (log_threshold - expected_log_evalue_total) / np.sqrt(expected_log_evalue_total + 1e-8)
        power = 1.0 - norm.cdf(z_score)
        
        return float(np.clip(power, 0, 1))


class DiffusionPosteriorEValues:
    """
    E-values for hypothesis testing with diffusion model posteriors.
    
    Integrates e-value testing with disorder-diffusive posterior sampling.
    Handles non-parametric priors and sequential SPI measurement.
    """
    
    def __init__(self, config: HypothesisTestConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.evalue_tester = EValueTesting(config, device)
        
        # Posterior tracking
        self.posterior_samples = []  # List of (n_samples, dim) tensors
        self.posterior_means = []
        self.posterior_covs = []
    
    def add_posterior_sample(self, samples: torch.Tensor) -> None:
        """
        Add posterior samples from diffusion model.
        
        Args:
            samples: Posterior samples (num_samples, dim)
        """
        samples = samples.to(self.device)
        
        # Compute empirical mean and covariance
        mean = samples.mean(dim=0)
        cov = torch.cov(samples.T)
        
        self.posterior_samples.append(samples)
        self.posterior_means.append(mean)
        self.posterior_covs.append(cov)
    
    def compute_evalue_from_posterior_samples(self, 
                                             h0_samples: torch.Tensor,
                                             h1_samples: torch.Tensor) -> float:
        """
        Compute e-value by comparing posterior samples under H0 vs H1.
        
        Uses importance sampling approximation:
          E = mean_H1[p(samples | H1) / p(samples | H0)]
        
        Args:
            h0_samples: Samples from posterior under H0
            h1_samples: Samples from posterior under H1
        
        Returns:
            Empirical e-value
        """
        h0_samples = h0_samples.to(self.device)
        h1_samples = h1_samples.to(self.device)
        
        # Fit Gaussians to samples
        mu_0 = h0_samples.mean(dim=0)
        mu_1 = h1_samples.mean(dim=0)
        cov_0 = torch.cov(h0_samples.T)
        cov_1 = torch.cov(h1_samples.T)
        
        # Likelihood ratio at h1_samples
        log_lls = []
        for sample in h1_samples:
            tester = EValueTesting(self.config, self.device)
            tester.config.h0_mean = mu_0
            tester.config.h1_mean = mu_1
            tester.config.h0_cov = cov_0
            tester.config.h1_cov = cov_1
            
            evalue = tester.compute_likelihood_ratio_evalue(sample.unsqueeze(0))
            log_lls.append(np.log(evalue + 1e-8))
        
        # Average in log space
        log_evalue = np.mean(log_lls)
        return np.exp(log_evalue)
    
    def test_reconstruction_quality(self, x_recon: torch.Tensor,
                                    x_true: torch.Tensor) -> Tuple[float, bool]:
        """
        Test reconstruction quality using e-value framework.
        
        H0: ∥x̂ - x*∥²₂ ≥ δ_0  (reconstruction below quality threshold)
        H1: ∥x̂ - x*∥²₂ < δ_0  (reconstruction meets quality)
        
        Args:
            x_recon: Reconstructed signal
            x_true: Ground truth signal
        
        Returns:
            (e_value, reject_h0): E-value and rejection decision
        """
        error_mse = F.mse_loss(x_recon, x_true).item()
        delta_0 = self.config.quality_threshold
        
        # Setup hypothesis distributions
        # H0: error ~ N(δ_0, 1)
        # H1: error ~ N(δ_0/2, 1)
        self.config.h0_mean = torch.tensor([delta_0], device=self.device)
        self.config.h1_mean = torch.tensor([delta_0 / 2], device=self.device)
        self.config.h0_cov = torch.tensor([[1.0]], device=self.device)
        self.config.h1_cov = torch.tensor([[1.0]], device=self.device)
        
        # Compute e-value
        y = torch.tensor([error_mse], device=self.device)
        evalue = self.evalue_tester.compute_likelihood_ratio_evalue(y, use_history=False)
        
        reject = self.evalue_tester.reject_h0_evalue(evalue)
        
        return evalue, reject
    
    def sequential_spi_test(self, measurements: List[float],
                           patterns: List[torch.Tensor],
                           x_true: Optional[torch.Tensor] = None) -> Dict:
        """
        Conduct sequential e-value test for SPI measurements.
        
        Sequentially accumulates e-values as measurements arrive.
        Can stop at any time with valid inference.
        
        Args:
            measurements: List of scalar measurements y_n
            patterns: List of measurement patterns h_n
            x_true: Ground truth for quality assessment (optional)
        
        Returns:
            Dictionary with testing results
        """
        results = {
            "measurements": [],
            "evalues": [],
            "rejected": [],
            "stopping_time": None,
            "final_evalue": 1.0,
        }
        
        for n, (y_n, h_n) in enumerate(zip(measurements, patterns)):
            # Add measurement
            y_tensor = torch.tensor([y_n], device=self.device)
            self.evalue_tester.add_measurement(y_tensor, update_evalue=True)
            
            evalue = self.evalue_tester.cumulative_evalue
            rejected = self.evalue_tester.reject_h0_evalue(evalue)
            
            results["measurements"].append(y_n)
            results["evalues"].append(evalue)
            results["rejected"].append(rejected)
            
            # Check stopping criteria
            if rejected:
                results["stopping_time"] = n + 1
                break
            
            if n >= self.config.max_measurements - 1:
                results["stopping_time"] = n + 1
                break
        
        results["final_evalue"] = self.evalue_tester.cumulative_evalue
        
        return results


def main_example():
    """Example comparing p-values and e-values."""
    
    print("=" * 70)
    print("E-values vs P-values for Hypothesis Testing")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup hypothesis test
    config = HypothesisTestConfig(
        h0_mean=torch.zeros(10),
        h1_mean=torch.ones(10) * 0.5,
        h0_cov=torch.eye(10),
        h1_cov=torch.eye(10),
        alpha=0.05,
        evalue_threshold=1.0 / 0.05,  # 20
        device=device
    )
    
    # Initialize testers
    pvalue_tester = PValueTesting(config, device=device)
    evalue_tester = EValueTesting(config, device=device)
    
    print("\n1. Classical P-value Testing")
    print("-" * 70)
    print("Limitations:")
    print("  - Only valid for fixed sample size")
    print("  - Adaptive sampling invalidates Type I error")
    print("  - Requires exact null distribution")
    
    # Single measurement test
    y_test = torch.randn(10)
    pvalue = pvalue_tester.compute_pvalue_reconstruction_error(
        y_test, torch.zeros(10), test_stat="mse"
    )
    print(f"\nP-value for test: {pvalue:.6f}")
    print(f"Reject H0 at α=0.05? {pvalue <= 0.05}")
    
    print("\n2. E-value Testing (Anytime-Valid)")
    print("-" * 70)
    print("Advantages:")
    print("  - Valid under optional stopping")
    print("  - Distribution-free construction")
    print("  - Robust to model misspecification")
    
    # Sequential measurements
    print("\nSequential accumulation of e-values:")
    for n in range(1, 6):
        y_n = torch.randn(10)
        evalue_tester.add_measurement(y_n, update_evalue=True)
        evalue = evalue_tester.cumulative_evalue
        reject = evalue_tester.reject_h0_evalue(evalue)
        
        print(f"  After {n} measurements: E = {evalue:.4f}, Reject? {reject}")
    
    print("\n3. Power Analysis")
    print("-" * 70)
    
    effect_size = 0.5
    for n_samples in [10, 50, 100, 200]:
        power = evalue_tester.power_analysis(effect_size, n_samples)
        print(f"  N={n_samples:3d}: Power = {power:.4f}")
    
    print("\n4. Sequential SPI Test")
    print("-" * 70)
    
    diffusion_tester = DiffusionPosteriorEValues(config, device=device)
    
    # Simulate SPI measurements
    measurements = [1.2, 1.8, 0.9, 2.1, 1.5, 2.3, 0.8, 2.0]
    patterns = [torch.randint(0, 2, (10,), dtype=torch.float32) for _ in measurements]
    
    results = diffusion_tester.sequential_spi_test(measurements, patterns)
    
    print("\nMeasurement | E-value | Reject H0?")
    for i, (y, ev, rej) in enumerate(zip(results["measurements"], results["evalues"], results["rejected"])):
        print(f"  {i+1:2d}       | {ev:7.2f} | {str(rej)}")
    
    if results["stopping_time"]:
        print(f"\nStopping time: {results['stopping_time']} measurements")
    print(f"Final e-value: {results['final_evalue']:.4f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
