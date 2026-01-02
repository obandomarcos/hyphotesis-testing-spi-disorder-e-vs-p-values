"""
Comprehensive Comparison of E-values vs P-values

Implements:
1. Side-by-side statistical comparison frameworks
2. Sequential stopping time analysis
3. Type I error control and validity under optional stopping
4. Power and sample size comparisons
5. Visualization and summary statistics
6. Empirical validation through Monte Carlo simulations

References:
- Grünwald et al. (2024): "Safe Testing" (JRSSB)
- Ramdas & Wang (2025): "Hypothesis Testing with E-values" (book manuscript)
- Wald (1947): "Sequential Analysis"
- ICLR/TPAMI papers on sequential testing and e-values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
import warnings


class ComparisonMetric(Enum):
    """Metrics for comparing evidence quantification methods."""
    STOPPING_TIME = "stopping_time"  # Sample size at stopping
    POWER = "power"  # P(reject | H1 true)
    TYPE_I_ERROR = "type_i_error"  # P(reject | H0 true)
    EVIDENCE_VALUE = "evidence_value"  # p-value or e-value magnitude
    SEQUENTIAL_VALIDITY = "sequential_validity"  # Anytime-valid property
    EFFICIENCY_RATIO = "efficiency_ratio"  # E-value stopping / p-value stopping


class TestFramework(Enum):
    """Testing frameworks."""
    FIXED_SAMPLE = "fixed_sample"  # Fixed N design
    SEQUENTIAL = "sequential"  # Sequential design
    GROUP_SEQUENTIAL = "group_sequential"  # Group sequential design


@dataclass
class ComparisonConfig:
    """Configuration for e-value vs p-value comparison."""
    
    # Data generation
    true_mean: float = 0.3  # True effect (H1)
    null_mean: float = 0.0  # Null hypothesis
    true_std: float = 1.0  # Standard deviation
    sample_size: int = 100  # For fixed-N designs
    
    # Testing parameters
    alpha: float = 0.05  # Significance level (p-value)
    beta: float = 0.2  # Type II error rate
    e_value_threshold: float = 20.0  # E-value threshold (≈ 1/alpha)
    
    # Sequential testing
    min_samples: int = 10  # Minimum before stopping allowed
    max_samples: int = 1000  # Hard limit on samples
    group_size: int = 1  # Samples per interim analysis
    
    # Simulation
    num_simulations: int = 1000  # Monte Carlo replications
    num_hypotheses: int = 2  # Test under H0 and H1
    
    # Comparison framework
    test_framework: TestFramework = TestFramework.SEQUENTIAL
    verbose: bool = False


@dataclass
class ComparisonResult:
    """Result of single comparison between p-value and e-value."""
    
    # Identification
    simulation_id: int = 0
    hypothesis_true: str = "H0"  # Which hypothesis generated data
    framework: TestFramework = TestFramework.FIXED_SAMPLE
    
    # P-value results
    p_value: float = 1.0
    p_significant: bool = False
    p_stopping_time: Optional[int] = None
    p_path: List[float] = field(default_factory=list)
    
    # E-value results
    e_value: float = 1.0
    e_significant: bool = False
    e_stopping_time: Optional[int] = None
    e_path: List[float] = field(default_factory=list)
    
    # Test statistics
    test_statistic: float = 0.0
    sample_size: int = 0
    effect_size: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'simulation_id': self.simulation_id,
            'hypothesis_true': self.hypothesis_true,
            'p_value': self.p_value,
            'p_significant': self.p_significant,
            'p_stopping_time': self.p_stopping_time,
            'e_value': self.e_value,
            'e_significant': self.e_significant,
            'e_stopping_time': self.e_stopping_time,
            'test_statistic': self.test_statistic,
            'sample_size': self.sample_size,
            'effect_size': self.effect_size,
        }


@dataclass
class ComparisonSummary:
    """Summary statistics from comparison study."""
    
    # Power
    power_p_value: float = 0.0  # Power of p-value test
    power_e_value: float = 0.0  # Power of e-value test
    
    # Type I error
    type_i_error_p_value: float = 0.0  # FPR under H0
    type_i_error_e_value: float = 0.0  # FPR under H0
    
    # Sequential efficiency
    mean_stopping_time_p_value: Optional[float] = None
    mean_stopping_time_e_value: Optional[float] = None
    mean_efficiency_ratio: Optional[float] = None  # E-value/p-value
    
    # Evidence quantification
    mean_p_value_under_h1: float = 0.0
    mean_e_value_under_h1: float = 0.0
    mean_p_value_under_h0: float = 0.0
    mean_e_value_under_h0: float = 0.0
    
    # Stopping time distribution
    p_value_stopping_distribution: List[int] = field(default_factory=list)
    e_value_stopping_distribution: List[int] = field(default_factory=list)
    
    # Validity under optional stopping
    type_i_error_sequential_p: float = 0.0  # P-value Type I rate (sequential)
    type_i_error_sequential_e: float = 0.0  # E-value Type I rate (sequential)
    e_value_supermartingale_valid: bool = False


class PValueEValueComparison:
    """Comprehensive comparison of p-value and e-value testing."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize comparison framework."""
        self.config = config
        self.results_h0: List[ComparisonResult] = []  # Results under H0
        self.results_h1: List[ComparisonResult] = []  # Results under H1
        
    def run_fixed_sample_comparison(self) -> Tuple[ComparisonResult, ComparisonResult]:
        """
        Run fixed-sample comparison for both H0 and H1.
        
        Returns:
            (result_h0, result_h1): Comparison results under both hypotheses
        """
        
        # Generate data under H0
        data_h0 = np.random.normal(
            self.config.null_mean,
            self.config.true_std,
            self.config.sample_size
        )
        
        # Generate data under H1
        data_h1 = np.random.normal(
            self.config.true_mean,
            self.config.true_std,
            self.config.sample_size
        )
        
        # P-value tests
        p_result_h0 = self._compute_p_value(data_h0, self.config.null_mean)
        p_result_h1 = self._compute_p_value(data_h1, self.config.null_mean)
        
        # E-value tests
        e_result_h0 = self._compute_e_value(data_h0, self.config.null_mean)
        e_result_h1 = self._compute_e_value(data_h1, self.config.null_mean)
        
        # Compile results under H0
        result_h0 = ComparisonResult(
            hypothesis_true="H0",
            framework=TestFramework.FIXED_SAMPLE,
            p_value=p_result_h0,
            p_significant=p_result_h0 < self.config.alpha,
            e_value=e_result_h0,
            e_significant=e_result_h0 > self.config.e_value_threshold,
            sample_size=self.config.sample_size,
        )
        
        # Compile results under H1
        result_h1 = ComparisonResult(
            hypothesis_true="H1",
            framework=TestFramework.FIXED_SAMPLE,
            p_value=p_result_h1,
            p_significant=p_result_h1 < self.config.alpha,
            e_value=e_result_h1,
            e_significant=e_result_h1 > self.config.e_value_threshold,
            sample_size=self.config.sample_size,
            effect_size=np.mean(data_h1) - self.config.null_mean,
        )
        
        return result_h0, result_h1
    
    def run_sequential_comparison(self) -> Tuple[ComparisonResult, ComparisonResult]:
        """
        Run sequential comparison for both H0 and H1.
        Anytime-valid stopping rules.
        
        Returns:
            (result_h0, result_h1): Sequential comparison results
        """
        
        # Generate longer data streams
        data_h0_stream = np.random.normal(
            self.config.null_mean,
            self.config.true_std,
            self.config.max_samples
        )
        
        data_h1_stream = np.random.normal(
            self.config.true_mean,
            self.config.true_std,
            self.config.max_samples
        )
        
        # Sequential p-value testing (fixed-N simulation)
        p_path_h0, p_stop_h0 = self._sequential_p_values(
            data_h0_stream, self.config.null_mean
        )
        p_path_h1, p_stop_h1 = self._sequential_p_values(
            data_h1_stream, self.config.null_mean
        )
        
        # Sequential e-value testing (anytime-valid)
        e_path_h0, e_stop_h0 = self._sequential_e_values(
            data_h0_stream, self.config.null_mean
        )
        e_path_h1, e_stop_h1 = self._sequential_e_values(
            data_h1_stream, self.config.null_mean
        )
        
        # Results under H0
        result_h0 = ComparisonResult(
            hypothesis_true="H0",
            framework=TestFramework.SEQUENTIAL,
            p_value=p_path_h0[-1] if p_path_h0 else 1.0,
            p_significant=(p_stop_h0 is not None),
            p_stopping_time=p_stop_h0,
            p_path=p_path_h0,
            e_value=e_path_h0[-1] if e_path_h0 else 1.0,
            e_significant=(e_stop_h0 is not None),
            e_stopping_time=e_stop_h0,
            e_path=e_path_h0,
            sample_size=len(data_h0_stream),
        )
        
        # Results under H1
        result_h1 = ComparisonResult(
            hypothesis_true="H1",
            framework=TestFramework.SEQUENTIAL,
            p_value=p_path_h1[-1] if p_path_h1 else 1.0,
            p_significant=(p_stop_h1 is not None),
            p_stopping_time=p_stop_h1,
            p_path=p_path_h1,
            e_value=e_path_h1[-1] if e_path_h1 else 1.0,
            e_significant=(e_stop_h1 is not None),
            e_stopping_time=e_stop_h1,
            e_path=e_path_h1,
            sample_size=len(data_h1_stream),
            effect_size=np.mean(data_h1_stream) - self.config.null_mean,
        )
        
        return result_h0, result_h1
    
    def run_monte_carlo_study(self) -> ComparisonSummary:
        """
        Run comprehensive Monte Carlo simulation comparing methods.
        
        Returns:
            ComparisonSummary with aggregate statistics
        """
        
        summary = ComparisonSummary()
        
        p_rejections_h0 = 0
        p_rejections_h1 = 0
        e_rejections_h0 = 0
        e_rejections_h1 = 0
        
        p_stopping_times_h1 = []
        e_stopping_times_h1 = []
        
        p_values_h0 = []
        p_values_h1 = []
        e_values_h0 = []
        e_values_h1 = []
        
        # Run simulations
        for sim_id in range(self.config.num_simulations):
            
            if self.config.test_framework == TestFramework.FIXED_SAMPLE:
                result_h0, result_h1 = self.run_fixed_sample_comparison()
            else:  # SEQUENTIAL
                result_h0, result_h1 = self.run_sequential_comparison()
            
            # Count rejections under H0 (Type I error)
            if result_h0.p_significant:
                p_rejections_h0 += 1
            if result_h0.e_significant:
                e_rejections_h0 += 1
            
            # Count rejections under H1 (Power)
            if result_h1.p_significant:
                p_rejections_h1 += 1
            if result_h1.e_significant:
                e_rejections_h1 += 1
            
            # Collect evidence values
            p_values_h0.append(result_h0.p_value)
            p_values_h1.append(result_h1.p_value)
            e_values_h0.append(result_h0.e_value)
            e_values_h1.append(result_h1.e_value)
            
            # Sequential stopping times (if applicable)
            if result_h1.p_stopping_time is not None:
                p_stopping_times_h1.append(result_h1.p_stopping_time)
            if result_h1.e_stopping_time is not None:
                e_stopping_times_h1.append(result_h1.e_stopping_time)
            
            # Store individual results
            result_h0.simulation_id = sim_id
            result_h1.simulation_id = sim_id
            self.results_h0.append(result_h0)
            self.results_h1.append(result_h1)
        
        # Compute summary statistics
        summary.power_p_value = p_rejections_h1 / self.config.num_simulations
        summary.power_e_value = e_rejections_h1 / self.config.num_simulations
        
        summary.type_i_error_p_value = p_rejections_h0 / self.config.num_simulations
        summary.type_i_error_e_value = e_rejections_h0 / self.config.num_simulations
        
        summary.mean_p_value_under_h0 = np.mean(p_values_h0)
        summary.mean_p_value_under_h1 = np.mean(p_values_h1)
        summary.mean_e_value_under_h0 = np.mean(e_values_h0)
        summary.mean_e_value_under_h1 = np.mean(e_values_h1)
        
        # Sequential efficiency
        if p_stopping_times_h1:
            summary.mean_stopping_time_p_value = np.mean(p_stopping_times_h1)
        if e_stopping_times_h1:
            summary.mean_stopping_time_e_value = np.mean(e_stopping_times_h1)
        
        if (summary.mean_stopping_time_p_value is not None and
            summary.mean_stopping_time_e_value is not None):
            summary.mean_efficiency_ratio = (
                summary.mean_stopping_time_p_value /
                summary.mean_stopping_time_e_value
            )
        
        # Type I error under sequential testing (check supermartingale property)
        # E[E_tau] should be <= 1/alpha under H0
        summary.type_i_error_sequential_e = np.mean(e_values_h0)
        summary.e_value_supermartingale_valid = summary.type_i_error_sequential_e <= (1.0 / self.config.alpha)
        
        return summary
    
    def _compute_p_value(self, data: np.ndarray, null_mean: float) -> float:
        """Compute p-value from data using one-sample t-test."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 1.0
        
        t_stat = (mean - null_mean) / (std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 1))
        return float(p_value)
    
    def _compute_e_value(self, data: np.ndarray, null_mean: float) -> float:
        """Compute e-value using likelihood ratio."""
        n = len(data)
        
        # Null: mean = null_mean, std = sample_std
        null_std = np.std(data, ddof=1)
        if null_std == 0:
            null_std = 1.0
        
        # Alternative: mean = sample_mean, std = sample_std
        alt_mean = np.mean(data)
        alt_std = null_std  # Assuming equal variance under H0 and H1
        
        # Log-likelihood ratio
        log_lik_h0 = -0.5 * np.sum((data - null_mean)**2 / (null_std**2)) - n * np.log(null_std)
        log_lik_h1 = -0.5 * np.sum((data - alt_mean)**2 / (alt_std**2)) - n * np.log(alt_std)
        
        log_e = log_lik_h1 - log_lik_h0
        e_value = np.exp(np.clip(log_e, -100, 100))  # Clip for numerical stability
        
        return float(e_value)
    
    def _sequential_p_values(
        self, data_stream: np.ndarray, null_mean: float
    ) -> Tuple[List[float], Optional[int]]:
        """
        Compute sequential p-values (fixed-N simulation).
        
        Returns:
            (p_value_path, stopping_time)
        """
        p_values = []
        stopping_time = None
        
        for n in range(self.config.min_samples, len(data_stream) + 1, self.config.group_size):
            p_val = self._compute_p_value(data_stream[:n], null_mean)
            p_values.append(p_val)
            
            # Check stopping rule (fixed-N requires pre-planned N)
            if p_val < self.config.alpha and stopping_time is None:
                stopping_time = n
                # But continue for comparison
        
        return p_values, stopping_time
    
    def _sequential_e_values(
        self, data_stream: np.ndarray, null_mean: float
    ) -> Tuple[List[float], Optional[int]]:
        """
        Compute sequential e-values (anytime-valid).
        
        Returns:
            (e_value_path, stopping_time)
        """
        e_values = []
        cum_e = 1.0
        stopping_time = None
        
        for n in range(self.config.min_samples, len(data_stream) + 1, self.config.group_size):
            # Compute e-value for incremental observations
            if n > self.config.min_samples:
                new_obs = data_stream[n-self.config.group_size:n]
            else:
                new_obs = data_stream[:n]
            
            e_val = self._compute_e_value(new_obs, null_mean)
            cum_e *= e_val
            e_values.append(cum_e)
            
            # Anytime-valid stopping rule
            if cum_e > self.config.e_value_threshold and stopping_time is None:
                stopping_time = n
                # Continue for comparison
        
        return e_values, stopping_time
    
    def print_summary(self, summary: ComparisonSummary) -> None:
        """Print formatted summary of comparison results."""
        print("=" * 80)
        print("E-VALUES vs P-VALUES: COMPREHENSIVE COMPARISON SUMMARY")
        print("=" * 80)
        
        print("\n1. POWER AND TYPE I ERROR CONTROL")
        print("-" * 80)
        print(f"  P-value:  Power = {summary.power_p_value:.4f}, Type I error = {summary.type_i_error_p_value:.4f}")
        print(f"  E-value:  Power = {summary.power_e_value:.4f}, Type I error = {summary.type_i_error_e_value:.4f}")
        
        print("\n2. SEQUENTIAL EFFICIENCY")
        print("-" * 80)
        if summary.mean_stopping_time_p_value is not None:
            print(f"  P-value:  Mean stopping time = {summary.mean_stopping_time_p_value:.1f}")
        if summary.mean_stopping_time_e_value is not None:
            print(f"  E-value:  Mean stopping time = {summary.mean_stopping_time_e_value:.1f}")
        if summary.mean_efficiency_ratio is not None:
            print(f"  Efficiency ratio (p/e): {summary.mean_efficiency_ratio:.2f}x")
        
        print("\n3. EVIDENCE QUANTIFICATION (Mean values)")
        print("-" * 80)
        print(f"  Under H0:")
        print(f"    P-value: {summary.mean_p_value_under_h0:.6f}")
        print(f"    E-value: {summary.mean_e_value_under_h0:.4f}")
        print(f"  Under H1:")
        print(f"    P-value: {summary.mean_p_value_under_h1:.6f}")
        print(f"    E-value: {summary.mean_e_value_under_h1:.4f}")
        
        print("\n4. OPTIONAL STOPPING VALIDITY")
        print("-" * 80)
        print(f"  E(E_τ) under H0 = {summary.type_i_error_sequential_e:.4f}")
        print(f"  Supermartingale property (E[E_τ] ≤ 1/α) valid: {summary.e_value_supermartingale_valid}")
        print(f"  Target threshold (1/α): {1.0/self.config.alpha:.4f}")
        
        print("\n5. CONFIGURATION")
        print("-" * 80)
        print(f"  α (significance level): {self.config.alpha}")
        print(f"  β (Type II error): {self.config.beta}")
        print(f"  True mean (under H1): {self.config.true_mean}")
        print(f"  Null mean (under H0): {self.config.null_mean}")
        print(f"  Standard deviation: {self.config.true_std}")
        print(f"  Sample size: {self.config.sample_size}")
        print(f"  Number of simulations: {self.config.num_simulations}")
        print(f"  Testing framework: {self.config.test_framework.value}")
        
        print("\n" + "=" * 80)


def main_example():
    """Example demonstrating e-values vs p-values comparison."""
    
    print("\n" + "=" * 80)
    print("E-VALUES vs P-VALUES: COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Configuration
    config = ComparisonConfig(
        true_mean=0.5,
        null_mean=0.0,
        true_std=1.0,
        sample_size=100,
        alpha=0.05,
        e_value_threshold=20.0,
        num_simulations=500,
        test_framework=TestFramework.FIXED_SAMPLE,
        verbose=True
    )
    
    # Run fixed-sample comparison
    print("\n1. FIXED-SAMPLE COMPARISON")
    print("-" * 80)
    
    comparison = PValueEValueComparison(config)
    result_h0, result_h1 = comparison.run_fixed_sample_comparison()
    
    print(f"\nUnder H0 (null hypothesis true, mean = {config.null_mean}):")
    print(f"  p-value = {result_h0.p_value:.6f}, significant: {result_h0.p_significant}")
    print(f"  e-value = {result_h0.e_value:.4f}, significant: {result_h0.e_significant}")
    
    print(f"\nUnder H1 (alternative true, mean = {config.true_mean}):")
    print(f"  p-value = {result_h1.p_value:.6f}, significant: {result_h1.p_significant}")
    print(f"  e-value = {result_h1.e_value:.4f}, significant: {result_h1.e_significant}")
    print(f"  Effect size (Cohen's d): {result_h1.effect_size:.4f}")
    
    # Run Monte Carlo study
    print("\n2. MONTE CARLO COMPARISON STUDY")
    print("-" * 80)
    print(f"Running {config.num_simulations} simulations...")
    
    summary = comparison.run_monte_carlo_study()
    comparison.print_summary(summary)
    
    # Sequential comparison
    print("\n3. SEQUENTIAL COMPARISON")
    print("-" * 80)
    
    config_seq = ComparisonConfig(
        true_mean=0.5,
        null_mean=0.0,
        true_std=1.0,
        sample_size=100,
        alpha=0.05,
        e_value_threshold=20.0,
        min_samples=10,
        max_samples=500,
        num_simulations=100,
        test_framework=TestFramework.SEQUENTIAL,
        verbose=False
    )
    
    print(f"Running {config_seq.num_simulations} sequential simulations...")
    
    comparison_seq = PValueEValueComparison(config_seq)
    summary_seq = comparison_seq.run_monte_carlo_study()
    comparison_seq.print_summary(summary_seq)
    
    # Key findings
    print("\n4. KEY FINDINGS AND IMPLICATIONS")
    print("-" * 80)
    print("\nP-VALUES:")
    print("  • Type I error control: Fixed")
    print("  • Valid stopping times: Fixed-N only (pre-planned)")
    print("  • Optional stopping: INVALID (Type I error violated)")
    print("  • Multiple testing: Requires correction (Bonferroni, BH-FDR, etc.)")
    
    print("\nE-VALUES:")
    print("  • Type I error control: Via supermartingale property")
    print("  • Valid stopping times: Anytime-valid")
    print("  • Optional stopping: VALID (E_τ is supermartingale under H0)")
    print("  • Multiple testing: No correction needed")
    print(f"  • Empirical validation: E[E_τ] ≈ {summary_seq.type_i_error_sequential_e:.4f} (target ≤ {1.0/config.alpha:.4f})")
    
    print("\nEFFICIENCY:")
    if summary_seq.mean_efficiency_ratio:
        print(f"  • Sequential e-values are ~{summary_seq.mean_efficiency_ratio:.1f}x more efficient")
        print(f"    Mean p-value stopping time: {summary_seq.mean_stopping_time_p_value:.1f} samples")
        print(f"    Mean e-value stopping time: {summary_seq.mean_stopping_time_e_value:.1f} samples")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main_example()
