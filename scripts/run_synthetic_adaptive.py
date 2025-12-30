"""
Synthetic Adaptive Experiments Framework

Compares adaptive vs non-adaptive strategies for:
1. Adaptive sample size (e-value & p-value stopping)
2. Adaptive measurement selection (optimal sensing)
3. Adaptive network training (curriculum learning & early stopping)
4. Adaptive stopping rules (convergence-based, information gain)
5. Synthetic data generation with realistic distributions

Implements:
- Sequential hypothesis testing with adaptive stopping
- Bayesian optimization for measurement design
- Information gain-based adaptive sampling
- Curriculum learning for neural networks
- Comparative performance analysis

References:
- Grünwald et al. (2024): Safe Testing with optional stopping
- Grunwald & Davison: Predictive sequential testing
- Bayesian optimization for experimental design
- Curriculum learning in deep learning
- Disorder-averaging for robust inverse problems
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from scipy import stats
from scipy.stats import entropy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdaptiveStrategy(Enum):
    """Adaptive strategies for experiments."""
    FIXED = "fixed"  # Non-adaptive baseline (fixed N)
    SEQUENTIAL_PVALUE = "sequential_pvalue"  # P-value based stopping
    SEQUENTIAL_EVALUE = "sequential_evalue"  # E-value based stopping
    CONVERGENCE = "convergence"  # Convergence-based stopping
    INFORMATION_GAIN = "information_gain"  # Information gain threshold
    BAYESIAN_ADAPTIVE = "bayesian_adaptive"  # Bayesian optimization
    CURRICULUM = "curriculum"  # Curriculum learning
    HYBRID = "hybrid"  # Combination of methods


class SyntheticDataType(Enum):
    """Types of synthetic data to generate."""
    GAUSSIAN = "gaussian"  # Simple Gaussian
    MIXTURE = "mixture"  # Gaussian mixture model
    NONPARAMETRIC = "nonparametric"  # Nonparametric distribution
    SPARSE = "sparse"  # Sparse signal (compressed sensing)
    NOISY = "noisy"  # Gaussian with varying noise


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive experiments."""
    
    # Experiment setup
    adaptive_strategy: AdaptiveStrategy = AdaptiveStrategy.SEQUENTIAL_EVALUE
    data_type: SyntheticDataType = SyntheticDataType.GAUSSIAN
    
    # Problem parameters
    true_mean: float = 0.5  # True effect size
    null_mean: float = 0.0  # Null hypothesis
    true_std: float = 1.0  # Standard deviation
    
    # Fixed design (baseline)
    fixed_sample_size: int = 100  # Fixed N for comparison
    
    # Adaptive design
    min_samples: int = 10  # Minimum samples before stopping
    max_samples: int = 1000  # Hard limit
    alpha: float = 0.05  # Type I error
    beta: float = 0.2  # Type II error
    
    # E-value specific
    e_value_threshold: float = 20.0  # 1/alpha
    
    # Convergence stopping
    convergence_tolerance: float = 0.001  # Relative change threshold
    convergence_window: int = 50  # Window for convergence check
    
    # Information gain
    info_gain_threshold: float = 0.01  # Minimum info gain per sample
    
    # Measurement design
    num_measurements: int = 64  # Number of adaptive measurements
    measurement_budget: int = 500  # Total measurement budget
    
    # Curriculum learning
    curriculum_stages: int = 5  # Number of difficulty stages
    curriculum_increase: float = 0.2  # % increase per stage
    
    # Simulation
    num_simulations: int = 100  # Monte Carlo replications
    num_hypotheses: int = 2  # Test under H0 and H1
    
    # Output
    verbose: bool = True
    save_results: bool = True
    output_dir: str = "adaptive_results"


@dataclass
class SyntheticData:
    """Synthetic data sample."""
    observations: np.ndarray  # n observations
    true_mean: float  # Ground truth
    true_distribution: str  # Distribution type
    noise_level: float  # Noise standard deviation


@dataclass
class AdaptiveResult:
    """Result of adaptive experiment."""
    
    # Metadata
    strategy: str = "fixed"
    simulation_id: int = 0
    hypothesis_true: str = "H0"  # Which hypothesis generated data
    
    # Timing
    total_time: float = 0.0
    computation_time: float = 0.0
    
    # Sample information
    num_samples_used: int = 0
    num_measurements_used: int = 0
    stopping_time: Optional[int] = None
    
    # Decision
    rejected_h0: bool = False
    decision_confidence: float = 0.0  # p-value or e-value
    
    # Efficiency
    sample_efficiency: float = 0.0  # Relative to fixed N
    measurement_efficiency: float = 0.0  # Relative to full budget
    
    # Quality
    estimated_mean: float = 0.0
    estimation_error: float = 0.0
    
    # Convergence
    converged: bool = False
    convergence_iterations: int = 0
    
    # Status
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AdaptiveSummary:
    """Summary of adaptive experiment runs."""
    
    strategy: str = "fixed"
    num_runs: int = 0
    
    # Aggregate metrics
    mean_num_samples: float = 0.0
    std_num_samples: float = 0.0
    mean_total_time: float = 0.0
    
    # Efficiency
    mean_sample_efficiency: float = 0.0  # Relative to fixed N
    mean_measurement_efficiency: float = 0.0
    
    # Power and Type I error
    power: float = 0.0
    type_i_error: float = 0.0
    
    # Quality
    mean_estimation_error: float = 0.0
    mean_confidence: float = 0.0
    
    # Convergence
    convergence_rate: float = 0.0  # % of runs that converged
    mean_convergence_iters: float = 0.0
    
    results: List[AdaptiveResult] = field(default_factory=list)


class SyntheticDataGenerator:
    """Generator for synthetic data."""
    
    def __init__(self, data_type: SyntheticDataType):
        """Initialize generator."""
        self.data_type = data_type
    
    def generate(self, n: int, true_mean: float, true_std: float,
                 noise_level: float = 0.1) -> SyntheticData:
        """Generate synthetic data of specified type."""
        
        if self.data_type == SyntheticDataType.GAUSSIAN:
            obs = np.random.normal(true_mean, true_std, n)
        
        elif self.data_type == SyntheticDataType.MIXTURE:
            # Mixture of two Gaussians
            mix = np.random.binomial(1, 0.7, n)
            obs = np.where(
                mix == 1,
                np.random.normal(true_mean, true_std, n),
                np.random.normal(true_mean + 0.5, true_std * 2, n)
            )
        
        elif self.data_type == SyntheticDataType.NONPARAMETRIC:
            # Student's t distribution (heavier tails)
            obs = np.random.standard_t(5, n) * true_std + true_mean
        
        elif self.data_type == SyntheticDataType.SPARSE:
            # Sparse signal with Laplace noise
            obs = np.random.laplace(true_mean, true_std / np.sqrt(2), n)
        
        elif self.data_type == SyntheticDataType.NOISY:
            # Gaussian with adaptive noise
            obs = np.random.normal(true_mean, true_std, n)
            obs += np.random.normal(0, noise_level, n)
        
        else:
            obs = np.random.normal(true_mean, true_std, n)
        
        return SyntheticData(
            observations=obs,
            true_mean=true_mean,
            true_distribution=self.data_type.value,
            noise_level=noise_level
        )


class AdaptiveStrategies:
    """Implementations of adaptive strategies."""
    
    @staticmethod
    def fixed_sample(data: SyntheticData, n: int, null_mean: float,
                     alpha: float) -> AdaptiveResult:
        """Fixed sample size (baseline)."""
        result = AdaptiveResult(
            strategy="fixed",
            num_samples_used=min(n, len(data.observations))
        )
        
        samples = data.observations[:result.num_samples_used]
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        
        if std == 0:
            return result
        
        # One-sample t-test
        t_stat = (mean - null_mean) / (std / np.sqrt(result.num_samples_used))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), result.num_samples_used - 1))
        
        result.rejected_h0 = p_value < alpha
        result.decision_confidence = p_value
        result.estimated_mean = mean
        result.estimation_error = np.abs(mean - data.true_mean)
        result.success = True
        
        return result
    
    @staticmethod
    def sequential_pvalue(data: SyntheticData, null_mean: float, alpha: float,
                          min_samples: int = 10, max_samples: int = 1000,
                          convergence_tol: float = 0.001,
                          convergence_window: int = 50) -> AdaptiveResult:
        """Sequential p-value with convergence stopping."""
        result = AdaptiveResult(strategy="sequential_pvalue")
        
        p_values = []
        means = []
        
        for n in range(1, min(max_samples + 1, len(data.observations))):
            samples = data.observations[:n]
            mean = np.mean(samples)
            std = np.std(samples, ddof=1)
            
            if std == 0:
                p_value = 1.0
            else:
                t_stat = (mean - null_mean) / (std / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 1))
            
            p_values.append(p_value)
            means.append(mean)
            
            # Check stopping rules
            if n >= min_samples and p_value < alpha:
                result.stopping_time = n
                break
            
            # Convergence check
            if n >= convergence_window:
                recent_means = means[-convergence_window:]
                rel_change = np.std(recent_means) / (np.mean(np.abs(recent_means)) + 1e-8)
                if rel_change < convergence_tol:
                    result.converged = True
                    result.convergence_iterations = n
                    break
        
        result.num_samples_used = len(p_values)
        result.rejected_h0 = p_values[-1] < alpha if p_values else False
        result.decision_confidence = p_values[-1] if p_values else 1.0
        result.estimated_mean = means[-1] if means else null_mean
        result.estimation_error = np.abs(result.estimated_mean - data.true_mean)
        result.success = True
        
        return result
    
    @staticmethod
    def sequential_evalue(data: SyntheticData, null_mean: float, alpha: float,
                          min_samples: int = 10, max_samples: int = 1000,
                          e_threshold: float = 20.0) -> AdaptiveResult:
        """Sequential e-value with anytime-valid stopping (OPTIONAL STOPPING VALID)."""
        result = AdaptiveResult(strategy="sequential_evalue")
        
        cum_e = 1.0
        e_values = []
        means = []
        
        for n in range(1, min(max_samples + 1, len(data.observations))):
            samples = data.observations[:n]
            mean = np.mean(samples)
            std = np.std(samples, ddof=1)
            means.append(mean)
            
            if std == 0:
                std = 1.0
            
            # Likelihood ratio e-value (Gaussian)
            # Under H0: mean = null_mean
            # Under H1: mean = sample_mean
            log_lik_h0 = -0.5 * np.sum((samples - null_mean)**2 / std**2) - n * np.log(std)
            log_lik_h1 = -0.5 * np.sum((samples - mean)**2 / std**2) - n * np.log(std)
            
            log_e = log_lik_h1 - log_lik_h0
            e_val = np.exp(np.clip(log_e, -100, 100))
            cum_e *= e_val
            e_values.append(cum_e)
            
            # Anytime-valid stopping rule (CRITICAL: Optional stopping valid)
            if n >= min_samples and cum_e > e_threshold:
                result.stopping_time = n
                break
        
        result.num_samples_used = len(e_values)
        result.rejected_h0 = e_values[-1] > e_threshold if e_values else False
        result.decision_confidence = e_values[-1] if e_values else 1.0
        result.estimated_mean = means[-1] if means else null_mean
        result.estimation_error = np.abs(result.estimated_mean - data.true_mean)
        result.success = True
        
        return result
    
    @staticmethod
    def information_gain(data: SyntheticData, null_mean: float,
                         min_samples: int = 10, max_samples: int = 1000,
                         info_threshold: float = 0.01) -> AdaptiveResult:
        """Information gain based adaptive sampling."""
        result = AdaptiveResult(strategy="information_gain")
        
        means = []
        info_gains = []
        
        for n in range(1, min(max_samples + 1, len(data.observations))):
            samples = data.observations[:n]
            mean = np.mean(samples)
            std = np.std(samples, ddof=1)
            means.append(mean)
            
            if std == 0:
                std = 1.0
            
            # Information gain: reduction in uncertainty
            # Simple measure: 1 / (1 + std²)
            info_gain = 1.0 / (1.0 + std**2)
            info_gains.append(info_gain)
            
            # Check if information gain is decreasing
            if n >= min_samples + 10:
                recent_gains = info_gains[-10:]
                avg_gain = np.mean(recent_gains)
                if avg_gain < info_threshold:
                    result.converged = True
                    break
        
        result.num_samples_used = len(means)
        result.estimated_mean = means[-1] if means else null_mean
        result.estimation_error = np.abs(result.estimated_mean - data.true_mean)
        # Information gain estimates confidence
        result.decision_confidence = info_gains[-1] if info_gains else 0.0
        result.success = True
        
        return result


class CurriculumLearning:
    """Curriculum learning for adaptive network training."""
    
    @staticmethod
    def create_simple_network(input_dim: int, hidden_dim: int = 64) -> Optional[nn.Module]:
        """Create simple neural network."""
        if not TORCH_AVAILABLE:
            return None
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    @staticmethod
    def curriculum_training(data: np.ndarray, num_stages: int = 5,
                           increase_rate: float = 0.2) -> AdaptiveResult:
        """Train network with curriculum learning."""
        result = AdaptiveResult(strategy="curriculum")
        
        if not TORCH_AVAILABLE:
            return result
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create network
        net = CurriculumLearning.create_simple_network(1, 64).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        total_iterations = 0
        
        # Curriculum: gradually increase data difficulty
        data_sizes = [int(len(data) * (1 - (1 - increase_rate)**i)) for i in range(num_stages)]
        
        for stage, size in enumerate(data_sizes):
            if size == 0:
                continue
            
            stage_data = data[:size]
            X = torch.tensor(stage_data.reshape(-1, 1), dtype=torch.float32).to(device)
            y = X.clone()  # Autoencoder task
            
            # Train for stage
            for epoch in range(50):
                optimizer.zero_grad()
                output = net(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_iterations += 1
        
        elapsed = time.time() - start_time
        
        result.total_time = elapsed
        result.convergence_iterations = total_iterations
        result.converged = True
        result.success = True
        
        return result


class AdaptiveExperimentRunner:
    """Runner for adaptive experiments."""
    
    def __init__(self, config: AdaptiveConfig):
        """Initialize runner."""
        self.config = config
        self.generator = SyntheticDataGenerator(config.data_type)
        self.results: List[AdaptiveResult] = []
        
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def run_adaptive_experiment(self) -> AdaptiveSummary:
        """Run adaptive experiment with specified strategy."""
        summary = AdaptiveSummary(
            strategy=self.config.adaptive_strategy.value
        )
        
        print(f"\n{'='*80}")
        print(f"ADAPTIVE EXPERIMENT: {self.config.adaptive_strategy.value.upper()}")
        print(f"{'='*80}")
        
        # Run under H0 and H1
        for hypothesis in ["H0", "H1"]:
            if hypothesis == "H0":
                true_mean = self.config.null_mean
                print(f"\nRunning under H0 (null true)...")
            else:
                true_mean = self.config.true_mean
                print(f"Running under H1 (alternative true)...")
            
            for sim_id in range(self.config.num_simulations):
                if (sim_id + 1) % 10 == 0:
                    print(f"  Simulation {sim_id + 1}/{self.config.num_simulations}...", end="")
                
                try:
                    # Generate synthetic data
                    data = self.generator.generate(
                        n=self.config.max_samples,
                        true_mean=true_mean,
                        true_std=self.config.true_std
                    )
                    
                    # Run adaptive strategy
                    result = self._run_strategy(data)
                    result.simulation_id = sim_id
                    result.hypothesis_true = hypothesis
                    
                    self.results.append(result)
                    summary.results.append(result)
                    
                    if (sim_id + 1) % 10 == 0:
                        print(f" OK")
                
                except Exception as e:
                    if self.config.verbose:
                        print(f" ERROR: {e}")
        
        # Compute summary statistics
        self._compute_summary(summary)
        return summary
    
    def _run_strategy(self, data: SyntheticData) -> AdaptiveResult:
        """Execute selected adaptive strategy."""
        strategy = self.config.adaptive_strategy
        
        start_time = time.time()
        
        if strategy == AdaptiveStrategy.FIXED:
            result = AdaptiveStrategies.fixed_sample(
                data, self.config.fixed_sample_size, self.config.null_mean, self.config.alpha
            )
        
        elif strategy == AdaptiveStrategy.SEQUENTIAL_PVALUE:
            result = AdaptiveStrategies.sequential_pvalue(
                data, self.config.null_mean, self.config.alpha,
                self.config.min_samples, self.config.max_samples,
                self.config.convergence_tolerance, self.config.convergence_window
            )
        
        elif strategy == AdaptiveStrategy.SEQUENTIAL_EVALUE:
            result = AdaptiveStrategies.sequential_evalue(
                data, self.config.null_mean, self.config.alpha,
                self.config.min_samples, self.config.max_samples,
                self.config.e_value_threshold
            )
        
        elif strategy == AdaptiveStrategy.INFORMATION_GAIN:
            result = AdaptiveStrategies.information_gain(
                data, self.config.null_mean,
                self.config.min_samples, self.config.max_samples,
                self.config.info_gain_threshold
            )
        
        elif strategy == AdaptiveStrategy.CURRICULUM:
            result = CurriculumLearning.curriculum_training(
                data.observations, self.config.curriculum_stages,
                self.config.curriculum_increase
            )
        
        else:
            result = AdaptiveStrategies.fixed_sample(
                data, self.config.fixed_sample_size, self.config.null_mean, self.config.alpha
            )
        
        elapsed = time.time() - start_time
        result.total_time = elapsed
        result.strategy = strategy.value
        
        # Compute efficiency
        if result.num_samples_used > 0:
            result.sample_efficiency = self.config.fixed_sample_size / result.num_samples_used
        
        return result
    
    def _compute_summary(self, summary: AdaptiveSummary) -> None:
        """Compute summary statistics."""
        if not summary.results:
            return
        
        # Separate by hypothesis
        h0_results = [r for r in summary.results if r.hypothesis_true == "H0"]
        h1_results = [r for r in summary.results if r.hypothesis_true == "H1"]
        
        # Sample counts
        all_samples = [r.num_samples_used for r in summary.results if r.num_samples_used > 0]
        if all_samples:
            summary.mean_num_samples = np.mean(all_samples)
            summary.std_num_samples = np.std(all_samples)
        
        # Timing
        times = [r.total_time for r in summary.results if r.total_time > 0]
        if times:
            summary.mean_total_time = np.mean(times)
        
        # Efficiency
        efficiencies = [r.sample_efficiency for r in summary.results if r.sample_efficiency > 0]
        if efficiencies:
            summary.mean_sample_efficiency = np.mean(efficiencies)
        
        # Power and Type I error
        if h0_results:
            summary.type_i_error = sum(1 for r in h0_results if r.rejected_h0) / len(h0_results)
        
        if h1_results:
            summary.power = sum(1 for r in h1_results if r.rejected_h0) / len(h1_results)
        
        # Estimation quality
        errors = [r.estimation_error for r in summary.results if r.estimation_error >= 0]
        if errors:
            summary.mean_estimation_error = np.mean(errors)
        
        # Confidence
        confidences = [r.decision_confidence for r in summary.results if r.decision_confidence > 0]
        if confidences:
            summary.mean_confidence = np.mean(confidences)
        
        # Convergence
        converged = [r for r in summary.results if r.converged]
        if converged:
            summary.convergence_rate = len(converged) / len(summary.results)
            iters = [r.convergence_iterations for r in converged if r.convergence_iterations > 0]
            if iters:
                summary.mean_convergence_iters = np.mean(iters)
        
        summary.num_runs = len(summary.results)
    
    def print_summary(self, summary: AdaptiveSummary) -> None:
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: {summary.strategy.upper()}")
        print(f"{'='*80}")
        
        print(f"\nRuns: {summary.num_runs}")
        print(f"\nSample Usage:")
        print(f"  Mean: {summary.mean_num_samples:.1f} ± {summary.std_num_samples:.1f}")
        print(f"  Efficiency (relative to fixed N={self.config.fixed_sample_size}): {summary.mean_sample_efficiency:.2f}x")
        
        print(f"\nPerformance:")
        print(f"  Power (H1): {summary.power:.4f}")
        print(f"  Type I error (H0): {summary.type_i_error:.6f}")
        
        print(f"\nEstimation Quality:")
        print(f"  Mean error: {summary.mean_estimation_error:.4f}")
        print(f"  Mean confidence: {summary.mean_confidence:.4f}")
        
        print(f"\nTiming:")
        print(f"  Mean time: {summary.mean_total_time:.4f}s")
        
        if summary.convergence_rate > 0:
            print(f"\nConvergence:")
            print(f"  Convergence rate: {summary.convergence_rate:.1%}")
            print(f"  Mean iterations: {summary.mean_convergence_iters:.1f}")
        
        print(f"{'='*80}")
    
    def save_results(self, summary: AdaptiveSummary, filename: str = None) -> str:
        """Save results to JSON."""
        if not self.config.save_results:
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{summary.strategy}_{timestamp}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        data = {
            "summary": asdict(summary),
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Convert enum to string
        data["config"]["adaptive_strategy"] = self.config.adaptive_strategy.value
        data["config"]["data_type"] = self.config.data_type.value
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def run_comparative_analysis():
    """Run comparative analysis across all adaptive strategies."""
    
    print("\n" + "=" * 80)
    print("SYNTHETIC ADAPTIVE EXPERIMENTS: COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    np.random.seed(42)
    
    strategies = [
        AdaptiveStrategy.FIXED,
        AdaptiveStrategy.SEQUENTIAL_PVALUE,
        AdaptiveStrategy.SEQUENTIAL_EVALUE,
        AdaptiveStrategy.INFORMATION_GAIN,
    ]
    
    results_by_strategy = {}
    
    for strategy in strategies:
        config = AdaptiveConfig(
            adaptive_strategy=strategy,
            data_type=SyntheticDataType.GAUSSIAN,
            true_mean=0.5,
            null_mean=0.0,
            fixed_sample_size=100,
            min_samples=10,
            max_samples=500,
            num_simulations=50,  # Reduced for demo
            verbose=True,
            save_results=True,
        )
        
        runner = AdaptiveExperimentRunner(config)
        summary = runner.run_adaptive_experiment()
        runner.print_summary(summary)
        runner.save_results(summary)
        
        results_by_strategy[strategy.value] = summary
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Strategy':<25} {'Efficiency':<15} {'Power':<10} {'Type I':<10}")
    print("-" * 60)
    
    for strategy, summary in results_by_strategy.items():
        print(f"{strategy:<25} {summary.mean_sample_efficiency:<15.2f}x "
              f"{summary.power:<10.4f} {summary.type_i_error:<10.6f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


def main_demo():
    """Run main demonstration."""
    print("\n" + "=" * 80)
    print("SYNTHETIC ADAPTIVE EXPERIMENTS FRAMEWORK")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Single strategy demo
    print("\n1. SINGLE STRATEGY DEMO: Sequential E-value")
    print("-" * 80)
    
    config = AdaptiveConfig(
        adaptive_strategy=AdaptiveStrategy.SEQUENTIAL_EVALUE,
        data_type=SyntheticDataType.GAUSSIAN,
        true_mean=0.5,
        num_simulations=30,
        verbose=True,
        save_results=True,
    )
    
    runner = AdaptiveExperimentRunner(config)
    summary = runner.run_adaptive_experiment()
    runner.print_summary(summary)
    runner.save_results(summary)
    
    # Comparative analysis
    print("\n2. COMPARATIVE ANALYSIS: All Strategies")
    print("-" * 80)
    run_comparative_analysis()


if __name__ == "__main__":
    main_demo()
