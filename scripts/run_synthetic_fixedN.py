"""
Fixed-N Synthetic Experiments Framework

Comprehensive fixed-sample-size experiments for:
1. Baseline performance analysis (no adaptation)
2. Power and Type I error calculation
3. Sample size planning (effect size sensitivity)
4. Effect size estimation and confidence intervals
5. Statistical efficiency comparison with adaptive methods
6. Robustness under distribution misspecification

Implements:
- Classical hypothesis testing (one-sample t-test, z-test)
- Multiple effect sizes and sample sizes
- Distribution sensitivity analysis
- Confidence interval construction
- Statistical power curves
- Sample size planning (n for target power)
- Comparison benchmarks for adaptive methods

References:
- Cohen (1988): Statistical Power Analysis
- Casella & Berger (2002): Statistical Inference
- Fleiss et al. (2003): Statistical Methods for Rates and Proportions
- CWI research on disorder-averaging and inverse problems
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from scipy import stats

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"  # One-sample t-test
    Z_TEST = "z_test"  # One-sample z-test (known variance)
    WILCOXON = "wilcoxon"  # Nonparametric test
    PAIRED_T = "paired_t"  # Paired t-test
    WELCH_T = "welch_t"  # Welch's t-test (unequal variance)


class DistributionType(Enum):
    """Types of distributions."""
    GAUSSIAN = "gaussian"  # N(mu, sigma²)
    STUDENT_T = "student_t"  # Heavy tails
    LAPLACE = "laplace"  # Sparse/compressed sensing
    MIXTURE = "mixture"  # Mixture of normals
    UNIFORM = "uniform"  # Bounded support


class HypothesisType(Enum):
    """Hypothesis test types."""
    ONE_SIDED_LEFT = "one_sided_left"  # H0: mu >= 0 vs H1: mu < 0
    ONE_SIDED_RIGHT = "one_sided_right"  # H0: mu <= 0 vs H1: mu > 0
    TWO_SIDED = "two_sided"  # H0: mu = 0 vs H1: mu != 0


@dataclass
class FixedNConfig:
    """Configuration for fixed-N experiments."""
    
    # Test setup
    test_type: TestType = TestType.T_TEST
    distribution_type: DistributionType = DistributionType.GAUSSIAN
    hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED
    
    # Problem parameters
    null_mean: float = 0.0  # H0: mean = null_mean
    true_means: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0])
    true_std: float = 1.0  # Population standard deviation
    
    # Sample sizes to test
    sample_sizes: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500, 1000])
    
    # Significance level
    alpha: float = 0.05  # Type I error
    desired_power: float = 0.80  # Target power for planning
    
    # Simulation
    num_simulations: int = 500  # Monte Carlo replications per condition
    num_replicates: int = 1  # Multiple experimental runs
    
    # Analysis
    compute_ci: bool = True  # Compute confidence intervals
    ci_level: float = 0.95  # Confidence level
    compute_power_curve: bool = True  # Compute power as function of N
    
    # Output
    verbose: bool = True
    save_results: bool = True
    save_plots: bool = True
    output_dir: str = "fixedN_results"


@dataclass
class FixedNResult:
    """Result of fixed-N test run."""
    
    # Metadata
    test_type: str = "t_test"
    distribution: str = "gaussian"
    hypothesis: str = "two_sided"
    replicate_id: int = 0
    
    # Problem specification
    null_mean: float = 0.0
    true_mean: float = 0.0
    true_std: float = 1.0
    sample_size: int = 100
    effect_size: float = 0.0  # Cohen's d
    
    # Test results
    test_statistic: float = 0.0
    p_value: float = 1.0
    rejected_h0: bool = False
    
    # Estimation
    estimated_mean: float = 0.0
    estimated_std: float = 0.0
    estimation_error: float = 0.0
    
    # Confidence interval
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ci_coverage: bool = False  # True if CI covers true mean
    
    # Efficiency
    standard_error: float = 0.0
    margin_of_error: float = 0.0
    
    # Status
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FixedNSummary:
    """Summary of fixed-N experiments."""
    
    test_type: str = "t_test"
    distribution: str = "gaussian"
    hypothesis: str = "two_sided"
    
    # Grouping variables
    true_mean: float = 0.0
    sample_size: int = 100
    effect_size: float = 0.0
    
    # Aggregate statistics
    num_simulations: int = 0
    power: float = 0.0  # P(reject | H1)
    type_i_error: float = 0.0  # P(reject | H0)
    
    # Estimation accuracy
    mean_p_value: float = 0.0
    std_p_value: float = 0.0
    mean_test_stat: float = 0.0
    
    # Confidence interval properties
    mean_ci_width: float = 0.0
    mean_ci_lower: float = 0.0
    mean_ci_upper: float = 0.0
    ci_coverage_rate: float = 0.0  # % of CIs containing true mean
    
    # Estimation accuracy
    mean_estimation_error: float = 0.0
    mean_standard_error: float = 0.0
    
    # Distribution
    p_value_distribution: List[float] = field(default_factory=list)
    test_stat_distribution: List[float] = field(default_factory=list)
    
    results: List[FixedNResult] = field(default_factory=list)


class DataGenerator:
    """Generator for synthetic data under various distributions."""
    
    def __init__(self, distribution: DistributionType):
        """Initialize generator."""
        self.distribution = distribution
    
    def generate(self, n: int, mean: float, std: float) -> np.ndarray:
        """Generate n samples from specified distribution."""
        
        if self.distribution == DistributionType.GAUSSIAN:
            return np.random.normal(mean, std, n)
        
        elif self.distribution == DistributionType.STUDENT_T:
            # t-distribution with 5 df, scaled to have std = std
            return np.random.standard_t(5, n) * (std / np.sqrt(5/(5-2))) + mean
        
        elif self.distribution == DistributionType.LAPLACE:
            # Laplace (double exponential) scaled to have std = std
            return np.random.laplace(mean, std / np.sqrt(2), n)
        
        elif self.distribution == DistributionType.MIXTURE:
            # Mixture: 70% N(mean, std), 30% N(mean+0.5*std, 2*std)
            mix = np.random.binomial(1, 0.7, n)
            return np.where(
                mix == 1,
                np.random.normal(mean, std, n),
                np.random.normal(mean + 0.5 * std, 2 * std, n)
            )
        
        elif self.distribution == DistributionType.UNIFORM:
            # Uniform on [mean - c*std, mean + c*std] where c = sqrt(3)
            width = 2 * np.sqrt(3) * std
            return np.random.uniform(mean - width/2, mean + width/2, n)
        
        else:
            return np.random.normal(mean, std, n)


class FixedNTester:
    """Performs fixed-N statistical tests."""
    
    def __init__(self, config: FixedNConfig):
        """Initialize tester."""
        self.config = config
        self.generator = DataGenerator(config.distribution_type)
    
    def run_test(self, data: np.ndarray, null_mean: float) -> Tuple[float, float, float]:
        """
        Run test and return (test_stat, p_value, test_type).
        
        Returns:
            (test_statistic, p_value, type: "parametric" or "nonparametric")
        """
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        if self.config.test_type == TestType.T_TEST:
            if sample_std == 0:
                return np.nan, 1.0, "parametric"
            
            t_stat = (sample_mean - null_mean) / (sample_std / np.sqrt(n))
            
            if self.config.hypothesis_type == HypothesisType.TWO_SIDED:
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 1))
            elif self.config.hypothesis_type == HypothesisType.ONE_SIDED_RIGHT:
                p_value = 1 - stats.t.cdf(t_stat, n - 1)
            else:  # ONE_SIDED_LEFT
                p_value = stats.t.cdf(t_stat, n - 1)
            
            return t_stat, p_value, "parametric"
        
        elif self.config.test_type == TestType.Z_TEST:
            # Z-test with known variance
            z_stat = (sample_mean - null_mean) / (self.config.true_std / np.sqrt(n))
            
            if self.config.hypothesis_type == HypothesisType.TWO_SIDED:
                p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            elif self.config.hypothesis_type == HypothesisType.ONE_SIDED_RIGHT:
                p_value = 1 - stats.norm.cdf(z_stat)
            else:  # ONE_SIDED_LEFT
                p_value = stats.norm.cdf(z_stat)
            
            return z_stat, p_value, "parametric"
        
        elif self.config.test_type == TestType.WILCOXON:
            # Wilcoxon signed-rank test (nonparametric)
            from scipy.stats import wilcoxon
            try:
                result = wilcoxon(data - null_mean, alternative='two-sided')
                return result.statistic, result.pvalue, "nonparametric"
            except Exception:
                return np.nan, 1.0, "nonparametric"
        
        elif self.config.test_type == TestType.WELCH_T:
            # Welch's t-test (does not assume equal variance)
            if sample_std == 0:
                return np.nan, 1.0, "parametric"
            
            t_stat = (sample_mean - null_mean) / (sample_std / np.sqrt(n))
            # Welch-Satterthwaite df
            df = (sample_std**2 / n)**2 / ((sample_std**2 / n)**2 / (n - 1))
            
            if self.config.hypothesis_type == HypothesisType.TWO_SIDED:
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            elif self.config.hypothesis_type == HypothesisType.ONE_SIDED_RIGHT:
                p_value = 1 - stats.t.cdf(t_stat, df)
            else:
                p_value = stats.t.cdf(t_stat, df)
            
            return t_stat, p_value, "parametric"
        
        else:
            return np.nan, 1.0, "parametric"
    
    def compute_ci(self, data: np.ndarray, null_mean: float,
                   ci_level: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for mean."""
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        if sample_std == 0 or n < 2:
            return sample_mean, sample_mean
        
        if self.config.test_type == TestType.Z_TEST:
            # Z-based CI
            se = self.config.true_std / np.sqrt(n)
            z_crit = stats.norm.ppf((1 + ci_level) / 2)
            margin = z_crit * se
        else:
            # t-based CI (default)
            se = sample_std / np.sqrt(n)
            t_crit = stats.t.ppf((1 + ci_level) / 2, n - 1)
            margin = t_crit * se
        
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin
        
        return ci_lower, ci_upper


class FixedNExperimentRunner:
    """Runner for fixed-N experiments."""
    
    def __init__(self, config: FixedNConfig):
        """Initialize runner."""
        self.config = config
        self.tester = FixedNTester(config)
        self.results: List[FixedNResult] = []
        
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def run_fixed_n_experiment(self) -> Dict[Tuple[float, int], FixedNSummary]:
        """
        Run comprehensive fixed-N experiments.
        
        Returns:
            Dictionary: (true_mean, sample_size) -> FixedNSummary
        """
        
        print(f"\n{'='*80}")
        print("FIXED-N SYNTHETIC EXPERIMENTS")
        print(f"{'='*80}")
        
        summaries = {}
        
        # Loop over all conditions
        for true_mean in self.config.true_means:
            for sample_size in self.config.sample_sizes:
                
                # Skip if effect is under H0
                if true_mean == self.config.null_mean:
                    hypothesis = "H0"
                else:
                    hypothesis = "H1"
                
                print(f"\nCondition: true_mean={true_mean:.2f}, N={sample_size}, "
                      f"({hypothesis}) ...", end="")
                
                summary = self._run_condition(true_mean, sample_size)
                summaries[(true_mean, sample_size)] = summary
                
                print(f" Power={summary.power:.3f}, Type I={summary.type_i_error:.4f}")
        
        return summaries
    
    def _run_condition(self, true_mean: float, sample_size: int) -> FixedNSummary:
        """Run single experimental condition."""
        
        summary = FixedNSummary(
            test_type=self.config.test_type.value,
            distribution=self.config.distribution_type.value,
            hypothesis=self.config.hypothesis_type.value,
            true_mean=true_mean,
            sample_size=sample_size,
        )
        
        rejections = 0
        p_values = []
        test_stats = []
        ci_widths = []
        ci_lower_bounds = []
        ci_upper_bounds = []
        ci_coverage = 0
        estimation_errors = []
        standard_errors = []
        
        for sim_id in range(self.config.num_simulations):
            
            # Generate data
            data = self.tester.generator.generate(
                sample_size, true_mean, self.config.true_std
            )
            
            # Run test
            test_stat, p_value, _ = self.tester.run_test(data, self.config.null_mean)
            
            # Compute CI
            ci_lower, ci_upper = self.tester.compute_ci(
                data, self.config.null_mean, self.config.ci_level
            )
            
            # Accumulate results
            result = FixedNResult(
                test_type=self.config.test_type.value,
                distribution=self.config.distribution_type.value,
                hypothesis=self.config.hypothesis_type.value,
                true_mean=true_mean,
                sample_size=sample_size,
                null_mean=self.config.null_mean,
                true_std=self.config.true_std,
                test_statistic=test_stat,
                p_value=p_value,
                rejected_h0=p_value < self.config.alpha,
                estimated_mean=np.mean(data),
                estimated_std=np.std(data, ddof=1),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_coverage=(ci_lower <= true_mean <= ci_upper),
                standard_error=np.std(data, ddof=1) / np.sqrt(sample_size),
                margin_of_error=np.abs(ci_upper - ci_lower) / 2,
            )
            
            result.effect_size = (true_mean - self.config.null_mean) / self.config.true_std
            result.estimation_error = np.abs(result.estimated_mean - true_mean)
            
            self.results.append(result)
            summary.results.append(result)
            
            if result.rejected_h0:
                rejections += 1
            
            p_values.append(p_value)
            test_stats.append(test_stat)
            ci_widths.append(ci_upper - ci_lower)
            ci_lower_bounds.append(ci_lower)
            ci_upper_bounds.append(ci_upper)
            
            if result.ci_coverage:
                ci_coverage += 1
            
            estimation_errors.append(result.estimation_error)
            standard_errors.append(result.standard_error)
        
        # Compute summary statistics
        summary.num_simulations = self.config.num_simulations
        
        # Power and Type I error
        if true_mean == self.config.null_mean:
            summary.type_i_error = rejections / self.config.num_simulations
            summary.power = 0.0
        else:
            summary.power = rejections / self.config.num_simulations
            summary.type_i_error = 0.0
        
        # P-value distribution
        summary.p_value_distribution = p_values
        summary.mean_p_value = np.mean(p_values)
        summary.std_p_value = np.std(p_values)
        
        # Test statistics
        summary.test_stat_distribution = test_stats
        summary.mean_test_stat = np.mean([t for t in test_stats if not np.isnan(t)])
        
        # Confidence intervals
        summary.mean_ci_width = np.mean(ci_widths)
        summary.mean_ci_lower = np.mean(ci_lower_bounds)
        summary.mean_ci_upper = np.mean(ci_upper_bounds)
        summary.ci_coverage_rate = ci_coverage / self.config.num_simulations
        
        # Estimation accuracy
        summary.mean_estimation_error = np.mean(estimation_errors)
        summary.mean_standard_error = np.mean(standard_errors)
        
        return summary
    
    def plot_power_curves(self, summaries: Dict[Tuple[float, int], FixedNSummary]) -> None:
        """Plot power curves as function of sample size."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by effect size
        effect_sizes = {}
        for (true_mean, sample_size), summary in summaries.items():
            effect = summary.effect_size
            if effect not in effect_sizes:
                effect_sizes[effect] = {"sizes": [], "power": []}
            effect_sizes[effect]["sizes"].append(sample_size)
            effect_sizes[effect]["power"].append(summary.power)
        
        # Plot power curves
        ax = axes[0]
        for effect, data in sorted(effect_sizes.items()):
            if effect > 0:  # Only plot non-null effects
                sizes = sorted(zip(data["sizes"], data["power"]))
                sizes, powers = zip(*sizes)
                ax.plot(sizes, powers, marker='o', label=f'd={effect:.2f}')
        
        ax.axhline(y=self.config.desired_power, color='r', linestyle='--', 
                   label=f'Target power ({self.config.desired_power})')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Power')
        ax.set_title('Statistical Power Curves')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot CI width vs sample size
        ax = axes[1]
        ci_widths_by_n = {}
        for (true_mean, sample_size), summary in summaries.items():
            if true_mean == self.config.null_mean:
                if sample_size not in ci_widths_by_n:
                    ci_widths_by_n[sample_size] = []
                ci_widths_by_n[sample_size].append(summary.mean_ci_width)
        
        sizes = sorted(ci_widths_by_n.keys())
        widths = [np.mean(ci_widths_by_n[s]) for s in sizes]
        
        ax.plot(sizes, widths, marker='s', color='green', linewidth=2)
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Mean CI Width')
        ax.set_title(f'Confidence Interval Width (CI level={self.config.ci_level})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            filepath = os.path.join(self.config.output_dir, "power_curves.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        plt.show()
    
    def print_summary_table(self, summaries: Dict[Tuple[float, int], FixedNSummary]) -> None:
        """Print summary table of results."""
        
        print(f"\n{'='*120}")
        print("FIXED-N RESULTS SUMMARY TABLE")
        print(f"{'='*120}")
        
        # H0 results
        print(f"\nTYPE I ERROR (under H0: true_mean = {self.config.null_mean})")
        print(f"{'N':<10} {'Type I Error':<20} {'Mean CI Width':<20} {'CI Coverage':<20}")
        print("-" * 70)
        
        h0_summaries = [(n, s) for (m, n), s in summaries.items() if m == self.config.null_mean]
        for sample_size, summary in sorted(h0_summaries):
            print(f"{sample_size:<10} {summary.type_i_error:<20.6f} "
                  f"{summary.mean_ci_width:<20.4f} {summary.ci_coverage_rate:<20.3f}")
        
        # H1 results (power)
        print(f"\nSTATISTICAL POWER (as function of effect size and N)")
        print(f"{'N':<10} {'Effect Size':<15} {'Power':<15} {'Est. Error':<15}")
        print("-" * 55)
        
        h1_summaries = [(n, m, s) for (m, n), s in summaries.items() if m != self.config.null_mean]
        for sample_size, true_mean, summary in sorted(h1_summaries, key=lambda x: (x[0], x[1])):
            print(f"{sample_size:<10} {summary.effect_size:<15.2f} {summary.power:<15.4f} "
                  f"{summary.mean_estimation_error:<15.4f}")
        
        print(f"{'='*120}")
    
    def save_results(self, summaries: Dict[Tuple[float, int], FixedNSummary],
                     filename: str = None) -> str:
        """Save results to JSON."""
        
        if not self.config.save_results:
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fixedN_{timestamp}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Convert summaries to serializable format
        summary_data = {}
        for (true_mean, sample_size), summary in summaries.items():
            key = f"effect_{summary.effect_size:.2f}_N_{sample_size}"
            summary_data[key] = {
                "true_mean": true_mean,
                "sample_size": sample_size,
                "effect_size": summary.effect_size,
                "power": summary.power,
                "type_i_error": summary.type_i_error,
                "mean_p_value": summary.mean_p_value,
                "ci_coverage_rate": summary.ci_coverage_rate,
                "mean_ci_width": summary.mean_ci_width,
            }
        
        data = {
            "summaries": summary_data,
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Convert enums to strings
        data["config"]["test_type"] = self.config.test_type.value
        data["config"]["distribution_type"] = self.config.distribution_type.value
        data["config"]["hypothesis_type"] = self.config.hypothesis_type.value
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main_demo():
    """Run main demonstration."""
    
    print("\n" + "=" * 80)
    print("FIXED-N SYNTHETIC EXPERIMENTS")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Configuration
    config = FixedNConfig(
        test_type=TestType.T_TEST,
        distribution_type=DistributionType.GAUSSIAN,
        hypothesis_type=HypothesisType.TWO_SIDED,
        null_mean=0.0,
        true_means=[0.0, 0.2, 0.5, 0.8],  # H0 and various effect sizes
        true_std=1.0,
        sample_sizes=[25, 50, 100, 200, 500],  # Multiple sample sizes
        alpha=0.05,
        desired_power=0.80,
        num_simulations=500,
        compute_ci=True,
        compute_power_curve=True,
        verbose=True,
        save_results=True,
        save_plots=MATPLOTLIB_AVAILABLE,
    )
    
    # Run experiments
    runner = FixedNExperimentRunner(config)
    summaries = runner.run_fixed_n_experiment()
    
    # Print summary
    runner.print_summary_table(summaries)
    
    # Plot power curves
    if config.compute_power_curve:
        runner.plot_power_curves(summaries)
    
    # Save results
    runner.save_results(summaries)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {config.output_dir}")
    print("Fixed-N provides baseline for adaptive comparison.")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main_demo()
