"""
E-values vs P-values for Sequential Hypothesis Testing

Implements:
  1. Classical p-value framework with limitations
  2. E-value framework with optional stopping validity
  3. Likelihood ratio e-values for Gaussian models
  4. Sequential testing procedures
  5. Comparison and empirical validation

References:
  - Grünwald et al. (2024): "Safe Testing" (JRSSB)
  - Ramdas & Wang (2025): "Hypothesis Testing with E-values" (book manuscript)
  - TPAMI/ICLR papers: Sequential testing sections
  - Classical hypothesis testing textbooks
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


class HypothesisType(Enum):
    """Types of hypothesis tests."""
    ONE_SAMPLE_MEAN = "one_sample_mean"
    TWO_SAMPLE_MEANS = "two_sample_means"
    VARIANCE_TEST = "variance_test"
    GOODNESS_OF_FIT = "goodness_of_fit"
    SEQUENTIAL_LR = "sequential_lr"


class EvidenceQuantification(Enum):
    """Evidence quantification methods."""
    P_VALUE = "p_value"
    E_VALUE = "e_value"
    BOTH = "both"


@dataclass
class HypothesisTestConfig:
    """Configuration for hypothesis testing."""
    # Hypothesis specification
    hypothesis_type: HypothesisType = HypothesisType.ONE_SAMPLE_MEAN
    null_hypothesis: str = "μ = μ0"
    alternative_hypothesis: str = "μ ≠ μ0"
    
    # Significance/evidence levels
    alpha: float = 0.05                # Type I error rate (p-value)
    beta: float = 0.2                  # Type II error rate
    e_value_threshold: float = 1.0/0.05  # E-value threshold for rejection (1/α)
    
    # Sequential testing
    is_sequential: bool = False        # Enable sequential testing
    max_samples: int = 1000            # Maximum samples in sequential design
    group_size: int = 1                # Samples per interim analysis
    
    # Effect size and power
    effect_size: float = 0.5           # Standardized effect size (Cohen's d)
    power: float = 0.8                 # Target power (1 - β)
    
    # Evidence quantification
    evidence_type: EvidenceQuantification = EvidenceQuantification.BOTH
    
    # Verbosity
    verbose: bool = False


@dataclass
class TestResult:
    """Result of hypothesis test."""
    # Test identification
    test_type: HypothesisType = HypothesisType.ONE_SAMPLE_MEAN
    
    # P-value results
    p_value: Optional[float] = None
    p_value_significant: Optional[bool] = None
    p_value_interpretation: str = ""
    
    # E-value results
    e_value: Optional[float] = None
    e_value_significant: Optional[bool] = None
    e_value_interpretation: str = ""
    
    # Test statistics
    test_statistic: float = 0.0
    degrees_of_freedom: Optional[int] = None
    
    # Sample information
    sample_size: int = 0
    effect_size: float = 0.0
    
    # Sequential testing
    is_sequential: bool = False
    stopping_time: Optional[int] = None
    
    # Assumptions
    assumptions_met: bool = True
    assumption_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_type': self.test_type.value,
            'p_value': self.p_value,
            'p_value_significant': self.p_value_significant,
            'e_value': self.e_value,
            'e_value_significant': self.e_value_significant,
            'test_statistic': self.test_statistic,
            'degrees_of_freedom': self.degrees_of_freedom,
            'sample_size': self.sample_size,
            'effect_size': self.effect_size,
            'stopping_time': self.stopping_time,
        }


class PValueTesting:
    """Classical p-value hypothesis testing."""
    
    @staticmethod
    def one_sample_t_test(data: np.ndarray, null_mean: float = 0.0,
                         alpha: float = 0.05,
                         alternative: str = "two-sided") -> TestResult:
        """
        One-sample t-test and p-value computation.
        
        H0: μ = μ0
        H1: μ ≠ μ0 (or μ > μ0, μ < μ0)
        
        Args:
            data: Sample data
            null_mean: Hypothesized mean
            alpha: Significance level
            alternative: "two-sided", "greater", "less"
        
        Returns:
            TestResult with p-value and decision
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        std_error = std / np.sqrt(n)
        
        # t-statistic
        t_stat = (mean - null_mean) / std_error
        df = n - 1
        
        # p-value based on alternative
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        else:  # less
            p_value = stats.t.cdf(t_stat, df)
        
        # Decision
        significant = p_value < alpha
        
        # Effect size
        d = (mean - null_mean) / std if std > 0 else 0
        
        return TestResult(
            test_type=HypothesisType.ONE_SAMPLE_MEAN,
            p_value=p_value,
            p_value_significant=significant,
            p_value_interpretation=f"p={p_value:.4f}, {'significant' if significant else 'not significant'} at α={alpha}",
            test_statistic=t_stat,
            degrees_of_freedom=df,
            sample_size=n,
            effect_size=d,
        )
    
    @staticmethod
    def two_sample_t_test(data1: np.ndarray, data2: np.ndarray,
                         alpha: float = 0.05,
                         alternative: str = "two-sided",
                         equal_var: bool = True) -> TestResult:
        """
        Two-sample t-test.
        
        H0: μ1 = μ2
        H1: μ1 ≠ μ2 (or μ1 > μ2, μ1 < μ2)
        
        Args:
            data1, data2: Sample data for groups
            alpha: Significance level
            alternative: "two-sided", "greater", "less"
            equal_var: Assume equal variances
        
        Returns:
            TestResult with p-value and decision
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        if equal_var:
            # Pooled standard error
            sp = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            se = sp * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(std1**2/n1 + std2**2/n2)
            df = (std1**2/n1 + std2**2/n2)**2 / (
                (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
            )
        
        # t-statistic
        t_stat = (mean1 - mean2) / se
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)
        
        # Decision
        significant = p_value < alpha
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return TestResult(
            test_type=HypothesisType.TWO_SAMPLE_MEANS,
            p_value=p_value,
            p_value_significant=significant,
            p_value_interpretation=f"p={p_value:.4f}, {'significant' if significant else 'not significant'} at α={alpha}",
            test_statistic=t_stat,
            degrees_of_freedom=int(df),
            sample_size=n1 + n2,
            effect_size=d,
        )


class EValueTesting:
    """E-value hypothesis testing framework."""
    
    @staticmethod
    def likelihood_ratio_e_value(data: np.ndarray, null_model: Callable,
                                alternative_model: Callable,
                                alpha: float = 0.05) -> TestResult:
        """
        Compute likelihood ratio e-value.
        
        E = LR = p(data | H1) / p(data | H0)
        
        Args:
            data: Observed data
            null_model: Log likelihood function under H0
            alternative_model: Log likelihood function under H1
            alpha: Significance level
        
        Returns:
            TestResult with e-value
        """
        # Log likelihoods
        log_lik_h0 = null_model(data)
        log_lik_h1 = alternative_model(data)
        
        # Log e-value (log LR)
        log_e = log_lik_h1 - log_lik_h0
        e_value = np.exp(log_e)
        
        # Threshold for rejection
        threshold = 1.0 / alpha
        significant = e_value > threshold
        
        interpretation = (
            f"E={e_value:.4f}, "
            f"log(E)={log_e:.4f}, "
            f"{'evidence for H1' if significant else 'insufficient evidence'} "
            f"(threshold={threshold:.2f})"
        )
        
        return TestResult(
            e_value=e_value,
            e_value_significant=significant,
            e_value_interpretation=interpretation,
            test_statistic=log_e,
        )
    
    @staticmethod
    def gaussian_e_value(data: np.ndarray, null_mean: float, null_std: float,
                        alt_mean: float, alt_std: float,
                        alpha: float = 0.05) -> TestResult:
        """
        E-value for comparing Gaussian hypotheses.
        
        H0: N(μ0, σ0²)
        H1: N(μ1, σ1²)
        
        E = (σ0/σ1)^n × exp(-½[x^T(Σ1^-1 - Σ0^-1)x + (μ0^T Σ0^-1 μ0 - μ1^T Σ1^-1 μ1)n])
        
        Args:
            data: Observed data
            null_mean, null_std: Parameters under H0
            alt_mean, alt_std: Parameters under H1
            alpha: Significance level
        
        Returns:
            TestResult with e-value
        """
        n = len(data)
        
        # Log-likelihood ratio
        log_lik_h0 = -0.5 * np.sum((data - null_mean)**2 / (null_std**2)) - n * np.log(null_std)
        log_lik_h1 = -0.5 * np.sum((data - alt_mean)**2 / (alt_std**2)) - n * np.log(alt_std)
        
        log_e = log_lik_h1 - log_lik_h0
        e_value = np.exp(log_e)
        
        # Threshold
        threshold = 1.0 / alpha
        significant = e_value > threshold
        
        interpretation = (
            f"E={e_value:.4f}, "
            f"{'reject H0' if significant else 'fail to reject H0'} "
            f"(threshold={threshold:.2f})"
        )
        
        return TestResult(
            e_value=e_value,
            e_value_significant=significant,
            e_value_interpretation=interpretation,
            test_statistic=log_e,
            sample_size=n,
        )
    
    @staticmethod
    def anytime_valid_e_value_process(data_stream: np.ndarray,
                                      null_model: Callable,
                                      alternative_model: Callable,
                                      alpha: float = 0.05) -> Tuple[List[float], Optional[int]]:
        """
        Compute cumulative e-value over data stream (anytime-valid).
        
        E_n = ∏_{i=1}^n p(y_i | H1) / p(y_i | H0)
        
        Args:
            data_stream: Sequential observations (n_observations,)
            null_model: Log likelihood function under H0
            alternative_model: Log likelihood function under H1
            alpha: Significance level for stopping
        
        Returns:
            (e_values, stopping_time): Cumulative e-values and optional stopping time
        """
        threshold = 1.0 / alpha
        e_values = []
        cum_e = 1.0
        stopping_time = None
        
        for t, obs in enumerate(data_stream):
            # Likelihood ratio for this observation
            log_lik_h0 = null_model(np.array([obs]))
            log_lik_h1 = alternative_model(np.array([obs]))
            lr_t = np.exp(log_lik_h1 - log_lik_h0)
            
            # Cumulative e-value
            cum_e *= lr_t
            e_values.append(cum_e)
            
            # Check stopping rule
            if cum_e > threshold and stopping_time is None:
                stopping_time = t + 1
        
        return e_values, stopping_time


class HypothesisTestComparison:
    """Compare p-value and e-value procedures."""
    
    @staticmethod
    def compare_evidence_quantification(data: np.ndarray,
                                       null_mean: float,
                                       config: HypothesisTestConfig) -> Dict:
        """
        Compare p-value and e-value quantification.
        
        Args:
            data: Sample data
            null_mean: Null hypothesis mean
            config: HypothesisTestConfig
        
        Returns:
            Dictionary with both results
        """
        results = {}
        
        # P-value test
        if config.evidence_type in [EvidenceQuantification.P_VALUE, EvidenceQuantification.BOTH]:
            p_result = PValueTesting.one_sample_t_test(
                data, null_mean, config.alpha
            )
            results['p_value'] = p_result
            
            if config.verbose:
                print(f"P-value approach:")
                print(f"  p-value = {p_result.p_value:.6f}")
                print(f"  Significant at α={config.alpha}: {p_result.p_value_significant}")
        
        # E-value test
        if config.evidence_type in [EvidenceQuantification.E_VALUE, EvidenceQuantification.BOTH]:
            # Define models
            def null_model(x):
                mean = null_mean
                std = np.std(x)
                return -0.5 * np.sum((x - mean)**2 / (std**2)) - len(x) * np.log(std + 1e-8)
            
            def alt_model(x):
                mean = np.mean(x)
                std = np.std(x)
                return -0.5 * np.sum((x - mean)**2 / (std**2)) - len(x) * np.log(std + 1e-8)
            
            e_result = EValueTesting.likelihood_ratio_e_value(
                data, null_model, alt_model, config.alpha
            )
            results['e_value'] = e_result
            
            if config.verbose:
                print(f"\nE-value approach:")
                print(f"  e-value = {e_result.e_value:.6f}")
                print(f"  Threshold = {1.0/config.alpha:.2f}")
                print(f"  Significant: {e_result.e_value_significant}")
        
        return results
    
    @staticmethod
    def sequential_comparison(data_stream: np.ndarray,
                             null_mean: float,
                             config: HypothesisTestConfig) -> Dict:
        """
        Compare sequential p-value and e-value stopping times.
        
        Args:
            data_stream: Sequential observations
            null_mean: Null hypothesis mean
            config: HypothesisTestConfig
        
        Returns:
            Dictionary with stopping times and evidence paths
        """
        results = {'n': len(data_stream)}
        
        # P-value stopping (fixed-N simulation)
        p_results = []
        for n in range(1, len(data_stream) + 1):
            p_res = PValueTesting.one_sample_t_test(data_stream[:n], null_mean, config.alpha)
            p_results.append(p_res)
        
        # Find p-value stopping time
        p_stopping = None
        for n, res in enumerate(p_results):
            if res.p_value_significant:
                p_stopping = n + 1
                break
        
        results['p_value_stopping_time'] = p_stopping
        results['p_value_path'] = [r.p_value for r in p_results]
        
        # E-value stopping (anytime-valid)
        def null_model(x):
            std = np.std(x) if len(x) > 1 else 1.0
            return -0.5 * np.sum((x - null_mean)**2 / (std**2 + 1e-8))
        
        def alt_model(x):
            mean = np.mean(x)
            std = np.std(x) if len(x) > 1 else 1.0
            return -0.5 * np.sum((x - mean)**2 / (std**2 + 1e-8))
        
        e_values, e_stopping = EValueTesting.anytime_valid_e_value_process(
            data_stream, null_model, alt_model, config.alpha
        )
        
        results['e_value_stopping_time'] = e_stopping
        results['e_value_path'] = e_values
        
        if config.verbose:
            print(f"Sequential comparison:")
            print(f"  P-value stopping time: {p_stopping}")
            print(f"  E-value stopping time: {e_stopping}")
            if p_stopping is not None and e_stopping is not None:
                print(f"  E-value is {p_stopping/e_stopping:.2f}x faster (or slower)")
        
        return results


def main_example():
    """Example demonstrating e-values vs p-values."""
    
    print("=" * 70)
    print("E-values vs P-values for Hypothesis Testing")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Configuration
    config = HypothesisTestConfig(
        hypothesis_type=HypothesisType.ONE_SAMPLE_MEAN,
        alpha=0.05,
        is_sequential=True,
        max_samples=200,
        verbose=True
    )
    
    # Generate synthetic data
    print("\n1. Generate Synthetic Data")
    print("-" * 70)
    
    null_mean = 0.0
    true_mean = 0.3  # Effect present
    true_std = 1.0
    n_samples = 100
    
    data = np.random.normal(true_mean, true_std, n_samples)
    print(f"  Sample mean: {np.mean(data):.4f}")
    print(f"  Sample std: {np.std(data):.4f}")
    print(f"  Sample size: {len(data)}")
    
    # Fixed-sample comparison
    print("\n2. Fixed-Sample Comparison (P-value vs E-value)")
    print("-" * 70)
    
    comparison = HypothesisTestComparison.compare_evidence_quantification(
        data, null_mean, config
    )
    
    # Sequential comparison
    print("\n3. Sequential Comparison (Anytime-Valid E-values)")
    print("-" * 70)
    
    # Generate longer data stream
    data_stream = np.random.normal(true_mean, true_std, config.max_samples)
    
    seq_results = HypothesisTestComparison.sequential_comparison(
        data_stream, null_mean, config
    )
    
    print(f"\n  Total observations: {seq_results['n']}")
    if seq_results['p_value_stopping_time'] is not None:
        print(f"  P-value stopping time: {seq_results['p_value_stopping_time']}")
    else:
        print(f"  P-value: No rejection at N={seq_results['n']}")
    
    if seq_results['e_value_stopping_time'] is not None:
        print(f"  E-value stopping time: {seq_results['e_value_stopping_time']}")
    else:
        print(f"  E-value: No rejection at N={seq_results['n']}")
    
    # Optional stopping validity
    print("\n4. Optional Stopping Validity")
    print("-" * 70)
    
    print("  P-values: Invalid under optional stopping")
    print("    - Type I error guarantee violated")
    print("    - Requires fixed sample size or pre-planned group sequential design")
    
    print("\n  E-values: Valid under optional stopping")
    print("    - E_n forms a supermartingale under H0")
    print("    - Can stop anytime: E(E_τ) ≤ 1 under H0")
    print("    - No multiple testing correction needed")
    
    # Two-sample test
    print("\n5. Two-Sample Hypothesis Test")
    print("-" * 70)
    
    data1 = np.random.normal(0.0, 1.0, 50)
    data2 = np.random.normal(0.3, 1.0, 50)
    
    two_sample_result = PValueTesting.two_sample_t_test(data1, data2, config.alpha)
    
    print(f"  Group 1: mean={np.mean(data1):.4f}, std={np.std(data1):.4f}")
    print(f"  Group 2: mean={np.mean(data2):.4f}, std={np.std(data2):.4f}")
    print(f"  t-statistic: {two_sample_result.test_statistic:.4f}")
    print(f"  p-value: {two_sample_result.p_value:.4f}")
    print(f"  Significant: {two_sample_result.p_value_significant}")
    print(f"  Cohen's d: {two_sample_result.effect_size:.4f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
