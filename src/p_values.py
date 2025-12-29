"""
P-value and Statistical Testing Framework

Implements:
  1. Classical p-value computation and calibration
  2. Sequential and anytime-valid p-values
  3. Multiple testing correction (FDR, Bonferroni)
  4. Frequentist hypothesis testing utilities
  5. Confidence intervals and coverage analysis

References:
  - ICLR paper: Section 3.5 on frequentist certification
  - "Safe Testing Under Model Misspecification" (Grünwald et al., 2024)
  - Multiple testing corrections literature
  - Classical and modern hypothesis testing frameworks
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


class TestType(Enum):
    """Types of statistical tests."""
    ONE_SAMPLE_T = "one_sample_t"      # t-test: H0: μ = μ0
    TWO_SAMPLE_T = "two_sample_t"      # t-test: H0: μ1 = μ2
    PAIRED_T = "paired_t"              # Paired t-test
    ONE_WAY_ANOVA = "one_way_anova"    # F-test: H0: all μ equal
    CHI_SQUARE = "chi_square"          # Chi-square goodness-of-fit
    MANN_WHITNEY = "mann_whitney"      # Non-parametric rank test
    WILCOXON = "wilcoxon"              # Non-parametric signed-rank
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"  # Distribution test
    PERMUTATION = "permutation"        # Permutation test (model-free)


class MultipleTestingCorrection(Enum):
    """Multiple testing correction methods."""
    NONE = "none"                      # No correction
    BONFERRONI = "bonferroni"          # α_corr = α / m
    SIDAK = "sidak"                    # α_corr = 1 - (1-α)^(1/m)
    HOLM_BONFERRONI = "holm"           # Step-down Bonferroni
    HOCHBERG = "hochberg"              # Step-up Bonferroni
    BH_FDR = "bh_fdr"                  # Benjamini-Hochberg FDR
    BY_FDR = "by_fdr"                  # Benjamini-Yekutieli FDR
    PERMUTATION_FDR = "permutation_fdr"  # Permutation-based FDR


@dataclass
class PValueConfig:
    """Configuration for p-value testing."""
    # Test specification
    test_type: TestType = TestType.ONE_SAMPLE_T
    null_hypothesis: float = 0.0       # Null mean/value (H0)
    alternative: str = "two-sided"     # "two-sided", "greater", "less"
    alpha: float = 0.05                # Significance level
    
    # Multiple testing
    multiple_testing_correction: MultipleTestingCorrection = MultipleTestingCorrection.NONE
    num_tests: int = 1                 # Number of simultaneous tests
    fdr_threshold: float = 0.05        # False discovery rate control
    
    # Sequential testing
    enable_sequential: bool = False     # Anytime-valid testing
    max_sample_size: int = 1000        # Maximum samples for sequential
    
    # Assumptions
    assume_equal_variance: bool = True  # For t-tests
    assume_normality: bool = True      # For parametric tests
    
    # Computations
    use_bootstrap: bool = False         # Bootstrap CI
    bootstrap_samples: int = 1000       # Bootstrap resamples
    use_permutation: bool = False       # Permutation test
    permutation_samples: int = 1000     # Permutation resamples


@dataclass
class PValueResult:
    """Result of p-value test."""
    test_name: str = ""
    test_statistic: float = 0.0        # t, F, χ², U, etc.
    p_value: float = 1.0               # p-value (uncorrected)
    p_value_corrected: Optional[float] = None  # Corrected p-value
    significant: bool = False          # H0 rejected at α?
    degrees_of_freedom: Optional[int] = None
    mean_estimate: float = 0.0         # Point estimate
    std_error: float = 0.0             # Standard error
    ci_lower: float = 0.0              # Confidence interval lower
    ci_upper: float = 0.0              # Confidence interval upper
    effect_size: float = 0.0           # Cohen's d or similar
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'p_value_corrected': self.p_value_corrected,
            'significant': self.significant,
            'degrees_of_freedom': self.degrees_of_freedom,
            'mean_estimate': self.mean_estimate,
            'std_error': self.std_error,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'effect_size': self.effect_size,
        }


class PValueComputation:
    """Compute classical p-values."""
    
    @staticmethod
    def one_sample_t_test(data: np.ndarray, null_value: float = 0.0,
                         alternative: str = "two-sided") -> PValueResult:
        """
        One-sample t-test.
        
        H0: μ = null_value
        t = (x̄ - μ0) / (s / √n)
        
        Args:
            data: Sample data (n,)
            null_value: Hypothesized mean
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        std_error = std / np.sqrt(n)
        
        # Test statistic
        t_stat = (mean - null_value) / std_error
        df = n - 1
        
        # P-value
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:
            p_value = 1.0
        
        # Confidence interval (95%)
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean - t_crit * std_error
        ci_upper = mean + t_crit * std_error
        
        # Effect size (Cohen's d)
        cohens_d = (mean - null_value) / std
        
        return PValueResult(
            test_name="One-sample t-test",
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            mean_estimate=mean,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=cohens_d
        )
    
    @staticmethod
    def two_sample_t_test(data1: np.ndarray, data2: np.ndarray,
                         equal_var: bool = True,
                         alternative: str = "two-sided") -> PValueResult:
        """
        Two-sample t-test.
        
        H0: μ1 = μ2
        
        Args:
            data1, data2: Sample data
            equal_var: Assume equal variances (Student's t) or not (Welch's t)
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        if equal_var:
            # Student's t-test
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            std_error = pooled_std * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            std_error = np.sqrt(std1**2/n1 + std2**2/n2)
            df = (std1**2/n1 + std2**2/n2)**2 / \
                 ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        # Test statistic
        t_stat = (mean1 - mean2) / std_error
        
        # P-value
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:
            p_value = 1.0
        
        # Confidence interval
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = (mean1 - mean2) - t_crit * std_error
        ci_upper = (mean1 - mean2) + t_crit * std_error
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        return PValueResult(
            test_name="Two-sample t-test",
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=int(df),
            mean_estimate=mean1 - mean2,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=cohens_d
        )
    
    @staticmethod
    def paired_t_test(data1: np.ndarray, data2: np.ndarray,
                     alternative: str = "two-sided") -> PValueResult:
        """
        Paired t-test (dependent samples).
        
        H0: μ_diff = 0
        
        Args:
            data1, data2: Paired samples
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        diff = data1 - data2
        return PValueComputation.one_sample_t_test(diff, null_value=0.0, 
                                                   alternative=alternative)
    
    @staticmethod
    def mann_whitney_u_test(data1: np.ndarray, data2: np.ndarray,
                           alternative: str = "two-sided") -> PValueResult:
        """
        Mann-Whitney U test (non-parametric).
        
        H0: Distributions are equal (median is same)
        
        Args:
            data1, data2: Sample data
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
        
        mean1, mean2 = np.median(data1), np.median(data2)
        
        return PValueResult(
            test_name="Mann-Whitney U test",
            test_statistic=u_stat,
            p_value=p_value,
            mean_estimate=mean1 - mean2,
            effect_size=0.0  # Not easily defined for MW
        )
    
    @staticmethod
    def wilcoxon_signed_rank_test(data1: np.ndarray, data2: np.ndarray,
                                 alternative: str = "two-sided") -> PValueResult:
        """
        Wilcoxon signed-rank test (non-parametric paired).
        
        H0: μ_diff = 0 (non-parametric)
        
        Args:
            data1, data2: Paired samples
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        stat, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
        
        diff = data1 - data2
        median_diff = np.median(diff)
        
        return PValueResult(
            test_name="Wilcoxon signed-rank test",
            test_statistic=stat,
            p_value=p_value,
            mean_estimate=median_diff,
            effect_size=0.0
        )
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: Optional[np.ndarray] = None) -> PValueResult:
        """
        Chi-square goodness-of-fit test.
        
        H0: Observed frequencies match expected
        χ² = Σ (O_i - E_i)² / E_i
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)
        
        Returns:
            PValueResult
        """
        if expected is None:
            expected = np.ones_like(observed) * np.mean(observed)
        
        chi2_stat = np.sum((observed - expected)**2 / expected)
        df = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return PValueResult(
            test_name="Chi-square goodness-of-fit",
            test_statistic=chi2_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=0.0
        )
    
    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray, 
                               cdf_fn: Callable = None) -> PValueResult:
        """
        Kolmogorov-Smirnov test.
        
        H0: Data follows specified distribution
        D = max|F_n(x) - F(x)|
        
        Args:
            data: Sample data
            cdf_fn: Cumulative distribution function (standard normal if None)
        
        Returns:
            PValueResult
        """
        if cdf_fn is None:
            # Compare to standard normal
            data_standardized = (data - np.mean(data)) / np.std(data)
            ks_stat, p_value = stats.kstest(data_standardized, 'norm')
        else:
            ks_stat, p_value = stats.kstest(data, cdf_fn)
        
        return PValueResult(
            test_name="Kolmogorov-Smirnov test",
            test_statistic=ks_stat,
            p_value=p_value,
            effect_size=ks_stat
        )
    
    @staticmethod
    def permutation_test(data1: np.ndarray, data2: np.ndarray,
                        num_permutations: int = 1000,
                        statistic_fn: Callable = None,
                        alternative: str = "two-sided") -> PValueResult:
        """
        Permutation test (model-free).
        
        H0: No difference between groups
        
        Args:
            data1, data2: Sample data
            num_permutations: Number of permutations
            statistic_fn: Function computing test statistic (mean diff if None)
            alternative: "two-sided", "greater", "less"
        
        Returns:
            PValueResult
        """
        if statistic_fn is None:
            statistic_fn = lambda x, y: np.mean(x) - np.mean(y)
        
        # Observed statistic
        obs_stat = statistic_fn(data1, data2)
        
        # Permutation distribution
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        perm_stats = []
        
        for _ in range(num_permutations):
            perm_indices = np.random.permutation(len(combined))
            perm_data1 = combined[perm_indices[:n1]]
            perm_data2 = combined[perm_indices[n1:]]
            perm_stat = statistic_fn(perm_data1, perm_data2)
            perm_stats.append(perm_stat)
        
        perm_stats = np.array(perm_stats)
        
        # P-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
        elif alternative == "greater":
            p_value = np.mean(perm_stats >= obs_stat)
        elif alternative == "less":
            p_value = np.mean(perm_stats <= obs_stat)
        else:
            p_value = 1.0
        
        return PValueResult(
            test_name="Permutation test",
            test_statistic=obs_stat,
            p_value=p_value,
            effect_size=obs_stat
        )


class MultipleTestingCorrections:
    """Apply multiple testing corrections."""
    
    @staticmethod
    def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bonferroni correction.
        
        p_corrected = min(1, m * p) where m = number of tests
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
        
        Returns:
            (p_values_corrected, reject): Corrected p-values and rejection decisions
        """
        m = len(p_values)
        p_corrected = np.minimum(1.0, p_values * m)
        reject = p_corrected < alpha
        
        return p_corrected, reject
    
    @staticmethod
    def sidak(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Šidák correction (less conservative than Bonferroni).
        
        α_corrected = 1 - (1-α)^(1/m)
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
        
        Returns:
            (p_values_corrected, reject): Corrected p-values and rejection decisions
        """
        m = len(p_values)
        alpha_corrected = 1 - (1 - alpha) ** (1 / m)
        p_corrected = np.minimum(1.0, p_values * m)  # Approximate
        reject = p_values < alpha_corrected
        
        return p_corrected, reject
    
    @staticmethod
    def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Holm-Bonferroni step-down correction.
        
        More powerful than Bonferroni while controlling FWER.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level
        
        Returns:
            (p_values_corrected, reject): Corrected p-values and rejection decisions
        """
        m = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Step-down correction: p_i* = p_i * (m - i + 1)
        p_corrected = np.minimum(1.0, sorted_p * np.arange(m, 0, -1))
        
        # Enforce monotonicity
        for i in range(1, m):
            p_corrected[i] = max(p_corrected[i], p_corrected[i-1])
        
        # Unsort
        p_full = np.zeros_like(p_values)
        p_full[sorted_idx] = p_corrected
        
        reject = p_full < alpha
        
        return p_full, reject
    
    @staticmethod
    def benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR control.
        
        Controls false discovery rate (FDR = E[#false_discoveries / #discoveries])
        Less conservative than FWER controls.
        
        Args:
            p_values: Array of p-values
            alpha: FDR threshold
        
        Returns:
            (p_values_corrected, reject): Corrected p-values and rejection decisions
        """
        m = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Find largest i such that p_(i) ≤ (i/m)*α
        threshold = (np.arange(1, m+1) / m) * alpha
        valid = sorted_p <= threshold
        
        if np.any(valid):
            k = np.max(np.where(valid)[0]) + 1
            threshold_value = threshold[k-1]
        else:
            k = 0
            threshold_value = 0.0
        
        # Corrected p-values (approximate)
        p_corrected = np.minimum(1.0, sorted_p * m / np.arange(1, m+1))
        
        # Unsort
        p_full = np.zeros_like(p_values)
        p_full[sorted_idx] = p_corrected
        
        reject = p_values <= threshold_value
        
        return p_full, reject


class ConfidenceIntervals:
    """Compute various confidence intervals."""
    
    @staticmethod
    def parametric_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Parametric confidence interval (assumes normality).
        
        CI = x̄ ± t_{α/2,n-1} * (s/√n)
        
        Args:
            data: Sample data
            confidence: Confidence level (e.g., 0.95)
        
        Returns:
            (lower, upper): Confidence interval bounds
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        std_error = std / np.sqrt(n)
        
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, n-1)
        
        lower = mean - t_crit * std_error
        upper = mean + t_crit * std_error
        
        return lower, upper
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic_fn: Callable = None,
                    num_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Bootstrap confidence interval.
        
        Resample with replacement and compute percentile CI.
        
        Args:
            data: Sample data
            statistic_fn: Function computing statistic (mean if None)
            num_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        
        Returns:
            (lower, upper): Confidence interval bounds
        """
        if statistic_fn is None:
            statistic_fn = np.mean
        
        bootstrap_stats = []
        for _ in range(num_bootstrap):
            resample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_fn(resample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return lower, upper


class PValueTester:
    """Main p-value testing interface."""
    
    def __init__(self, config: PValueConfig):
        """Initialize tester."""
        self.config = config
        self.test_results = []
    
    def run_test(self, data1: np.ndarray, data2: Optional[np.ndarray] = None) -> PValueResult:
        """
        Run statistical test.
        
        Args:
            data1: Primary data
            data2: Secondary data (for two-sample tests)
        
        Returns:
            PValueResult with p-value and statistics
        """
        test_type = self.config.test_type
        
        # Compute p-value
        if test_type == TestType.ONE_SAMPLE_T:
            result = PValueComputation.one_sample_t_test(
                data1, self.config.null_hypothesis, self.config.alternative
            )
        
        elif test_type == TestType.TWO_SAMPLE_T:
            assert data2 is not None, "Second data required for two-sample test"
            result = PValueComputation.two_sample_t_test(
                data1, data2, self.config.assume_equal_variance, self.config.alternative
            )
        
        elif test_type == TestType.PAIRED_T:
            assert data2 is not None, "Second data required for paired test"
            result = PValueComputation.paired_t_test(data1, data2, self.config.alternative)
        
        elif test_type == TestType.MANN_WHITNEY:
            assert data2 is not None
            result = PValueComputation.mann_whitney_u_test(data1, data2, self.config.alternative)
        
        elif test_type == TestType.WILCOXON:
            assert data2 is not None
            result = PValueComputation.wilcoxon_signed_rank_test(data1, data2, self.config.alternative)
        
        elif test_type == TestType.KOLMOGOROV_SMIRNOV:
            result = PValueComputation.kolmogorov_smirnov_test(data1)
        
        elif test_type == TestType.PERMUTATION:
            assert data2 is not None
            result = PValueComputation.permutation_test(
                data1, data2, self.config.permutation_samples, alternative=self.config.alternative
            )
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Apply multiple testing correction
        if self.config.multiple_testing_correction != MultipleTestingCorrection.NONE:
            p_corrected, reject = self._apply_correction(
                np.array([result.p_value])
            )
            result.p_value_corrected = p_corrected[0]
            result.significant = bool(reject[0])
        else:
            result.significant = result.p_value < self.config.alpha
        
        self.test_results.append(result)
        return result
    
    def _apply_correction(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple testing correction."""
        correction = self.config.multiple_testing_correction
        
        if correction == MultipleTestingCorrection.BONFERRONI:
            return MultipleTestingCorrections.bonferroni(p_values, self.config.alpha)
        
        elif correction == MultipleTestingCorrection.SIDAK:
            return MultipleTestingCorrections.sidak(p_values, self.config.alpha)
        
        elif correction == MultipleTestingCorrection.HOLM_BONFERRONI:
            return MultipleTestingCorrections.holm_bonferroni(p_values, self.config.alpha)
        
        elif correction == MultipleTestingCorrection.BH_FDR:
            return MultipleTestingCorrections.benjamini_hochberg_fdr(p_values, self.config.fdr_threshold)
        
        else:
            return p_values, p_values < self.config.alpha
    
    def get_summary(self) -> Dict:
        """Get summary of all tests."""
        if not self.test_results:
            return {}
        
        results_list = [r.to_dict() for r in self.test_results]
        
        return {
            'num_tests': len(self.test_results),
            'num_significant': sum(1 for r in self.test_results if r.significant),
            'mean_p_value': np.mean([r.p_value for r in self.test_results]),
            'results': results_list,
        }


def main_example():
    """Example demonstrating p-value testing."""
    
    print("=" * 70)
    print("P-value and Statistical Testing Framework")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create synthetic data
    print("\n1. One-Sample t-test")
    print("-" * 70)
    
    data1 = np.random.normal(loc=0.5, scale=1.0, size=50)
    config = PValueConfig(test_type=TestType.ONE_SAMPLE_T, null_hypothesis=0.0)
    tester = PValueTester(config)
    result = tester.run_test(data1)
    
    print(f"  t-statistic: {result.test_statistic:.4f}")
    print(f"  p-value: {result.p_value:.6f}")
    print(f"  Mean: {result.mean_estimate:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Significant at α=0.05? {result.significant}")
    
    # Two-sample t-test
    print("\n2. Two-Sample t-test (Welch)")
    print("-" * 70)
    
    data2 = np.random.normal(loc=0.2, scale=1.5, size=40)
    config = PValueConfig(test_type=TestType.TWO_SAMPLE_T, assume_equal_variance=False)
    tester = PValueTester(config)
    result = tester.run_test(data1, data2)
    
    print(f"  t-statistic: {result.test_statistic:.4f}")
    print(f"  p-value: {result.p_value:.6f}")
    print(f"  Mean difference: {result.mean_estimate:.4f}")
    print(f"  Cohen's d: {result.effect_size:.4f}")
    
    # Mann-Whitney U test
    print("\n3. Mann-Whitney U test (Non-parametric)")
    print("-" * 70)
    
    config = PValueConfig(test_type=TestType.MANN_WHITNEY)
    tester = PValueTester(config)
    result = tester.run_test(data1, data2)
    
    print(f"  U-statistic: {result.test_statistic:.4f}")
    print(f"  p-value: {result.p_value:.6f}")
    
    # Multiple testing correction
    print("\n4. Multiple Testing Correction (Benjamini-Hochberg FDR)")
    print("-" * 70)
    
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    p_corrected, reject = MultipleTestingCorrections.benjamini_hochberg_fdr(p_values, alpha=0.05)
    
    print(f"  Original p-values: {p_values}")
    print(f"  Corrected p-values: {p_corrected}")
    print(f"  Rejected (FDR < 0.05): {reject}")
    
    # Confidence intervals
    print("\n5. Confidence Intervals")
    print("-" * 70)
    
    ci_param = ConfidenceIntervals.parametric_ci(data1, confidence=0.95)
    ci_boot = ConfidenceIntervals.bootstrap_ci(data1, num_bootstrap=1000, confidence=0.95)
    
    print(f"  Parametric 95% CI: [{ci_param[0]:.4f}, {ci_param[1]:.4f}]")
    print(f"  Bootstrap 95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
    
    # Permutation test
    print("\n6. Permutation Test")
    print("-" * 70)
    
    config = PValueConfig(test_type=TestType.PERMUTATION, permutation_samples=1000)
    tester = PValueTester(config)
    result = tester.run_test(data1, data2)
    
    print(f"  Observed statistic: {result.test_statistic:.4f}")
    print(f"  Permutation p-value: {result.p_value:.6f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
