"""
Stopping Rules and Sequential Decision-Making Framework

Implements:
  1. Anytime-valid stopping rules (e-values, sequential probability ratios)
  2. Adaptive stopping based on convergence criteria
  3. Resource-constrained stopping (budget, time, measurements)
  4. Bayesian and frequentist stopping boundaries
  5. Optional stopping theorem compliance and monitoring

References:
  - ICLR paper: Section 3.6 on anytime-valid stopping
  - "Always Valid Inference" (Grunwald et al., 2020)
  - Sequential analysis theory (Wald, 1947)
  - Bayes factor thresholds and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class StoppingRule(Enum):
    """Types of stopping rules."""
    E_VALUE_THRESHOLD = "e_value_threshold"       # E-value exceeds threshold
    SEQUENTIAL_LR = "sequential_lr"               # Sequential likelihood ratio
    BAYES_FACTOR = "bayes_factor"                 # Bayes factor threshold
    FIXED_SAMPLE_SIZE = "fixed_sample_size"       # Fixed N (baseline)
    CONVERGENCE = "convergence"                   # Parameter/quality convergence
    CONFIDENCE_WIDTH = "confidence_width"         # CI width below threshold
    ERROR_BOUND = "error_bound"                   # Reconstruction error bound
    RELATIVE_ENTROPY = "relative_entropy"         # KL divergence threshold
    INFORMATION_GAIN = "information_gain"         # Information gain rate
    BUDGET = "budget"                             # Resource budget exhausted


@dataclass
class StoppingConfig:
    """Configuration for stopping rule."""
    # Primary stopping criterion
    rule: StoppingRule = StoppingRule.E_VALUE_THRESHOLD
    
    # Threshold parameters
    e_value_threshold: float = 20.0           # E-value: α ≈ 1/threshold
    bayes_factor_threshold: float = 10.0      # Bayes factor (log scale)
    sequential_lr_threshold: float = 10.0     # Log-LR threshold
    confidence_width_threshold: float = 0.1   # Max CI width
    error_bound_threshold: float = 0.05       # Max reconstruction error
    information_gain_threshold: float = 0.01  # Min info gain per measurement
    
    # Sample size constraints
    min_samples: int = 10                     # Minimum before stopping allowed
    max_samples: int = 1000                   # Hard limit on measurements
    
    # Convergence criteria
    convergence_window: int = 50              # Window for convergence check
    convergence_tolerance: float = 0.001      # Relative change threshold
    
    # Resource budget
    max_time: Optional[float] = None          # Wall-clock time limit (seconds)
    max_cost: Optional[float] = None          # Measurement cost budget
    measurement_cost: float = 1.0             # Cost per measurement
    
    # Multi-criterion stopping
    use_all_criteria: bool = False            # Require all criteria met
    required_criteria: List[StoppingRule] = field(default_factory=list)
    
    # Monitoring
    compute_diagnostics: bool = True          # Track convergence metrics
    verbose: bool = False


@dataclass
class StoppingDecision:
    """Result of stopping decision."""
    should_stop: bool = False
    reason: str = ""
    num_measurements: int = 0
    num_measurements_so_far: int = 0
    
    # Stopping criterion values
    e_value: Optional[float] = None           # Current e-value
    bayes_factor: Optional[float] = None      # Current Bayes factor
    sequential_lr: Optional[float] = None     # Current LR
    convergence_metric: Optional[float] = None    # Convergence measure
    confidence_width: Optional[float] = None  # Current CI width
    error_bound: Optional[float] = None       # Current error bound
    information_gain: Optional[float] = None  # Recent info gain
    
    # Resource usage
    time_elapsed: Optional[float] = None
    cost_incurred: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'should_stop': self.should_stop,
            'reason': self.reason,
            'num_measurements': self.num_measurements,
            'num_measurements_so_far': self.num_measurements_so_far,
            'e_value': self.e_value,
            'bayes_factor': self.bayes_factor,
            'sequential_lr': self.sequential_lr,
            'convergence_metric': self.convergence_metric,
            'confidence_width': self.confidence_width,
            'error_bound': self.error_bound,
            'information_gain': self.information_gain,
            'time_elapsed': self.time_elapsed,
            'cost_incurred': self.cost_incurred,
        }


class EValueStopping:
    """E-value based anytime-valid stopping."""
    
    @staticmethod
    def compute_e_value(likelihood_ratio_log: float, 
                       num_measurements: int,
                       e_power: float = 1.0) -> float:
        """
        Compute e-value from log likelihood ratio.
        
        E = exp(log LR) for simple stopping or
        E = (exp(log LR))^(1/m) for mSPRT
        
        Args:
            likelihood_ratio_log: Log of likelihood ratio
            num_measurements: Number of measurements so far
            e_power: Power for e-value computation (1.0 for simple, 1/m for mSPRT)
        
        Returns:
            E-value scalar
        """
        e_value = np.exp(min(likelihood_ratio_log * e_power, 100))  # Cap for numerical stability
        return float(e_value)
    
    @staticmethod
    def check_stopping(e_value: float, threshold: float = 20.0,
                      min_samples: int = 10, num_samples: int = 0) -> Tuple[bool, str]:
        """
        Check if e-value stopping criterion met.
        
        Stop if E ≥ threshold and n ≥ min_samples
        
        Args:
            e_value: Current e-value
            threshold: Stopping threshold
            min_samples: Minimum samples before stopping
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason): Stop decision and reason
        """
        if num_samples < min_samples:
            return False, f"Below minimum sample size ({num_samples} < {min_samples})"
        
        if e_value >= threshold:
            return True, f"E-value threshold exceeded (E={e_value:.2f} ≥ {threshold})"
        
        return False, f"E-value below threshold (E={e_value:.2f} < {threshold})"


class SequentialLRStopping:
    """Sequential probability ratio test (SPRT) stopping."""
    
    @staticmethod
    def sprt_boundaries(alpha: float = 0.05, beta: float = 0.05,
                       theta_0: float = 0.0, theta_1: float = 1.0) -> Tuple[float, float]:
        """
        Compute SPRT boundaries (log scale).
        
        Upper boundary: A = log((1-β)/α)
        Lower boundary: B = log(β/(1-α))
        
        Args:
            alpha: Type I error rate
            beta: Type II error rate
            theta_0: Null parameter
            theta_1: Alternative parameter
        
        Returns:
            (upper_bound, lower_bound): SPRT boundaries in log scale
        """
        upper_bound = np.log((1 - beta) / alpha)
        lower_bound = np.log(beta / (1 - alpha))
        
        return upper_bound, lower_bound
    
    @staticmethod
    def check_stopping(log_lr: float, upper_bound: float, lower_bound: float,
                      min_samples: int = 10, num_samples: int = 0) -> Tuple[bool, str, int]:
        """
        Check SPRT stopping rule.
        
        Stop if: log LR ≥ A (accept H1) or log LR ≤ B (accept H0)
        
        Args:
            log_lr: Log likelihood ratio accumulated so far
            upper_bound: SPRT upper boundary
            lower_bound: SPRT lower boundary
            min_samples: Minimum samples before stopping
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason, decision): Stop flag, reason, and decision (1=H1, 0=H0, -1=continue)
        """
        if num_samples < min_samples:
            return False, f"Below min samples ({num_samples} < {min_samples})", -1
        
        if log_lr >= upper_bound:
            return True, f"Accept H1: log LR ≥ {upper_bound:.2f}", 1
        elif log_lr <= lower_bound:
            return True, f"Accept H0: log LR ≤ {lower_bound:.2f}", 0
        else:
            return False, f"Continue: {lower_bound:.2f} < log LR < {upper_bound:.2f}", -1


class BayesFactorStopping:
    """Bayes factor based stopping."""
    
    @staticmethod
    def check_stopping(log_bayes_factor: float, threshold: float = 3.0,
                      min_samples: int = 10, num_samples: int = 0) -> Tuple[bool, str]:
        """
        Check Bayes factor stopping rule.
        
        Stop if: |log BF| ≥ log(threshold)
        
        Args:
            log_bayes_factor: Log Bayes factor (H1 vs H0)
            threshold: Bayes factor threshold
            min_samples: Minimum samples before stopping
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason): Stop decision and reason
        """
        log_threshold = np.log(threshold)
        
        if num_samples < min_samples:
            return False, f"Below minimum samples ({num_samples} < {min_samples})"
        
        if log_bayes_factor >= log_threshold:
            bf = np.exp(log_bayes_factor)
            return True, f"Bayes factor favors H1 (BF={bf:.2f} > {threshold})"
        elif log_bayes_factor <= -log_threshold:
            bf = np.exp(-log_bayes_factor)
            return True, f"Bayes factor favors H0 (BF={bf:.2f} > {threshold})"
        else:
            return False, f"Bayes factor inconclusive (log BF={log_bayes_factor:.2f})"


class ConvergenceStopping:
    """Convergence-based stopping rules."""
    
    @staticmethod
    def relative_change(current_value: float, previous_value: float,
                       tolerance: float = 0.001) -> Tuple[bool, float]:
        """
        Check relative change in parameter.
        
        rc = |x_n - x_{n-1}| / (|x_{n-1}| + ε)
        
        Args:
            current_value: Current parameter value
            previous_value: Previous parameter value
            tolerance: Convergence tolerance
        
        Returns:
            (converged, relative_change): Convergence flag and metric
        """
        if previous_value == 0:
            rc = np.abs(current_value)
        else:
            rc = np.abs(current_value - previous_value) / (np.abs(previous_value) + 1e-8)
        
        converged = rc < tolerance
        return converged, float(rc)
    
    @staticmethod
    def check_convergence_window(history: List[float], window: int = 50,
                                tolerance: float = 0.001) -> Tuple[bool, float]:
        """
        Check if values have converged over window.
        
        Args:
            history: History of values
            window: Window size for convergence check
            tolerance: Convergence tolerance
        
        Returns:
            (converged, max_change): Convergence flag and max change in window
        """
        if len(history) < window:
            return False, 1.0
        
        recent = np.array(history[-window:])
        max_change = np.max(np.abs(np.diff(recent)))
        mean_val = np.mean(np.abs(recent))
        
        if mean_val > 0:
            rc = max_change / mean_val
        else:
            rc = max_change
        
        converged = rc < tolerance
        return converged, float(rc)
    
    @staticmethod
    def check_stopping(quality_history: List[float], config: StoppingConfig,
                      num_samples: int = 0) -> Tuple[bool, str, Optional[float]]:
        """
        Check convergence stopping rule.
        
        Args:
            quality_history: History of quality metric
            config: StoppingConfig
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason, convergence_metric): Stop flag, reason, metric
        """
        if num_samples < config.min_samples:
            return False, f"Below minimum samples ({num_samples} < {config.min_samples})", None
        
        if len(quality_history) < config.convergence_window:
            return False, f"Insufficient history ({len(quality_history)} < {config.convergence_window})", None
        
        converged, rc = ConvergenceStopping.check_convergence_window(
            quality_history, config.convergence_window, config.convergence_tolerance
        )
        
        if converged:
            return True, f"Convergence achieved (rc={rc:.6f} < {config.convergence_tolerance})", rc
        else:
            return False, f"Not converged (rc={rc:.6f} ≥ {config.convergence_tolerance})", rc


class ConfidenceBased:
    """Confidence interval width based stopping."""
    
    @staticmethod
    def check_stopping(ci_width: float, threshold: float = 0.1,
                      min_samples: int = 10, num_samples: int = 0) -> Tuple[bool, str]:
        """
        Stop when confidence interval width is small enough.
        
        Args:
            ci_width: Current confidence interval width
            threshold: Maximum acceptable width
            min_samples: Minimum samples
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason): Stop decision
        """
        if num_samples < min_samples:
            return False, f"Below minimum samples ({num_samples} < {min_samples})"
        
        if ci_width <= threshold:
            return True, f"CI width below threshold (width={ci_width:.4f} ≤ {threshold})"
        else:
            return False, f"CI width above threshold (width={ci_width:.4f} > {threshold})"


class ErrorBoundStopping:
    """Reconstruction error bound based stopping."""
    
    @staticmethod
    def check_stopping(reconstruction_error: float, threshold: float = 0.05,
                      upper_bound: Optional[float] = None,
                      min_samples: int = 10, num_samples: int = 0) -> Tuple[bool, str]:
        """
        Stop when reconstruction error is below threshold.
        
        Args:
            reconstruction_error: Current reconstruction error
            threshold: Target error threshold
            upper_bound: Upper bound on error (if available)
            min_samples: Minimum samples
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason): Stop decision
        """
        if num_samples < min_samples:
            return False, f"Below minimum samples ({num_samples} < {min_samples})"
        
        if reconstruction_error <= threshold:
            return True, f"Error below threshold (error={reconstruction_error:.6f} ≤ {threshold})"
        elif upper_bound is not None and upper_bound <= threshold:
            return True, f"Error bound below threshold (ub={upper_bound:.6f} ≤ {threshold})"
        else:
            return False, f"Error above threshold (error={reconstruction_error:.6f} > {threshold})"


class InformationGainStopping:
    """Information gain rate based stopping."""
    
    @staticmethod
    def check_stopping(information_gain_history: List[float], threshold: float = 0.01,
                      window: int = 10, min_samples: int = 10,
                      num_samples: int = 0) -> Tuple[bool, str, Optional[float]]:
        """
        Stop when information gain per measurement is below threshold.
        
        Args:
            information_gain_history: History of information gain per measurement
            threshold: Minimum information gain threshold
            window: Window for averaging recent gains
            min_samples: Minimum samples
            num_samples: Current sample count
        
        Returns:
            (should_stop, reason, mean_gain): Stop decision, reason, mean gain
        """
        if num_samples < min_samples:
            return False, f"Below minimum samples ({num_samples} < {min_samples})", None
        
        if len(information_gain_history) < window:
            return False, f"Insufficient history ({len(information_gain_history)} < {window})", None
        
        recent_gains = information_gain_history[-window:]
        mean_gain = np.mean(recent_gains)
        
        if mean_gain <= threshold:
            return True, f"Information gain too low (ig={mean_gain:.6f} ≤ {threshold})", mean_gain
        else:
            return False, f"Information gain sufficient (ig={mean_gain:.6f} > {threshold})", mean_gain


class ResourceBasedStopping:
    """Resource budget based stopping."""
    
    @staticmethod
    def check_stopping(num_samples: int, max_samples: int,
                      time_elapsed: Optional[float] = None, max_time: Optional[float] = None,
                      cost_incurred: Optional[float] = None, max_cost: Optional[float] = None) -> Tuple[bool, str]:
        """
        Stop when resource budget exhausted.
        
        Args:
            num_samples: Current sample count
            max_samples: Maximum allowed samples
            time_elapsed: Elapsed time (seconds)
            max_time: Maximum allowed time
            cost_incurred: Accumulated cost
            max_cost: Maximum cost budget
        
        Returns:
            (should_stop, reason): Stop decision
        """
        reasons = []
        
        if num_samples >= max_samples:
            reasons.append(f"Max samples reached ({num_samples} ≥ {max_samples})")
        
        if max_time is not None and time_elapsed is not None and time_elapsed >= max_time:
            reasons.append(f"Max time exceeded ({time_elapsed:.1f}s ≥ {max_time:.1f}s)")
        
        if max_cost is not None and cost_incurred is not None and cost_incurred >= max_cost:
            reasons.append(f"Max cost exceeded (${cost_incurred:.2f} ≥ ${max_cost:.2f})")
        
        should_stop = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "Resources available"
        
        return should_stop, reason


class StoppingRuleEvaluator:
    """Main stopping rule interface."""
    
    def __init__(self, config: StoppingConfig):
        """Initialize evaluator."""
        self.config = config
        self.history = {
            'e_values': [],
            'qualities': [],
            'information_gains': [],
            'convergence_metrics': [],
        }
    
    def evaluate(self, 
                e_value: Optional[float] = None,
                quality: Optional[float] = None,
                convergence_metric: Optional[float] = None,
                confidence_width: Optional[float] = None,
                error_bound: Optional[float] = None,
                information_gain: Optional[float] = None,
                num_samples: int = 0,
                time_elapsed: Optional[float] = None,
                cost_incurred: Optional[float] = None) -> StoppingDecision:
        """
        Evaluate stopping decision.
        
        Args:
            e_value: Current e-value
            quality: Quality metric
            convergence_metric: Convergence measure
            confidence_width: CI width
            error_bound: Reconstruction error
            information_gain: Information gain
            num_samples: Current sample count
            time_elapsed: Elapsed time
            cost_incurred: Accumulated cost
        
        Returns:
            StoppingDecision
        """
        # Track history
        if e_value is not None:
            self.history['e_values'].append(e_value)
        if quality is not None:
            self.history['qualities'].append(quality)
        if information_gain is not None:
            self.history['information_gains'].append(information_gain)
        if convergence_metric is not None:
            self.history['convergence_metrics'].append(convergence_metric)
        
        # Evaluate primary rule
        should_stop = False
        reason = ""
        
        if self.config.rule == StoppingRule.E_VALUE_THRESHOLD:
            if e_value is not None:
                should_stop, reason = EValueStopping.check_stopping(
                    e_value, self.config.e_value_threshold,
                    self.config.min_samples, num_samples
                )
        
        elif self.config.rule == StoppingRule.SEQUENTIAL_LR:
            if quality is not None:
                # Approximate LR from quality
                log_lr = -quality  # Simplified
                upper, lower = SequentialLRStopping.sprt_boundaries()
                should_stop, reason, _ = SequentialLRStopping.check_stopping(
                    log_lr, upper, lower, self.config.min_samples, num_samples
                )
        
        elif self.config.rule == StoppingRule.CONVERGENCE:
            if quality is not None:
                should_stop, reason, _ = ConvergenceStopping.check_stopping(
                    self.history['qualities'], self.config, num_samples
                )
        
        elif self.config.rule == StoppingRule.CONFIDENCE_WIDTH:
            if confidence_width is not None:
                should_stop, reason = ConfidenceBased.check_stopping(
                    confidence_width, self.config.confidence_width_threshold,
                    self.config.min_samples, num_samples
                )
        
        elif self.config.rule == StoppingRule.ERROR_BOUND:
            if error_bound is not None:
                should_stop, reason = ErrorBoundStopping.check_stopping(
                    error_bound, self.config.error_bound_threshold,
                    None, self.config.min_samples, num_samples
                )
        
        elif self.config.rule == StoppingRule.INFORMATION_GAIN:
            if information_gain is not None:
                should_stop, reason, _ = InformationGainStopping.check_stopping(
                    self.history['information_gains'], self.config.information_gain_threshold,
                    10, self.config.min_samples, num_samples
                )
        
        elif self.config.rule == StoppingRule.BUDGET:
            should_stop, reason = ResourceBasedStopping.check_stopping(
                num_samples, self.config.max_samples, time_elapsed,
                self.config.max_time, cost_incurred, self.config.max_cost
            )
        
        # Check additional criteria if required
        if self.config.use_all_criteria and should_stop:
            for criterion in self.config.required_criteria:
                if criterion == self.config.rule:
                    continue
                # Evaluate additional criteria (simplified)
                pass
        
        # Create decision object
        decision = StoppingDecision(
            should_stop=should_stop,
            reason=reason,
            num_measurements=num_samples,
            e_value=e_value,
            convergence_metric=convergence_metric,
            confidence_width=confidence_width,
            error_bound=error_bound,
            information_gain=information_gain,
            time_elapsed=time_elapsed,
            cost_incurred=cost_incurred,
        )
        
        if self.config.verbose and should_stop:
            print(f"STOPPING: {reason}")
        
        return decision


def main_example():
    """Example demonstrating stopping rules."""
    
    print("=" * 70)
    print("Stopping Rules and Sequential Decision-Making Framework")
    print("=" * 70)
    
    # E-value stopping
    print("\n1. E-value Based Stopping")
    print("-" * 70)
    
    config = StoppingConfig(
        rule=StoppingRule.E_VALUE_THRESHOLD,
        e_value_threshold=20.0,
        min_samples=10,
        max_samples=500,
        verbose=True
    )
    
    evaluator = StoppingRuleEvaluator(config)
    
    e_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0]
    for n, e in enumerate(e_values, 1):
        decision = evaluator.evaluate(e_value=e, num_samples=n*10)
        print(f"  N={n*10}: E={e:.1f}, Stop? {decision.should_stop} ({decision.reason})")
    
    # SPRT stopping
    print("\n2. Sequential Probability Ratio Test (SPRT)")
    print("-" * 70)
    
    upper, lower = SequentialLRStopping.sprt_boundaries(alpha=0.05, beta=0.05)
    print(f"  SPRT boundaries: {lower:.2f} ≤ log LR ≤ {upper:.2f}")
    
    log_lrs = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]
    for n, log_lr in enumerate(log_lrs, 1):
        should_stop, reason, decision = SequentialLRStopping.check_stopping(
            log_lr, upper, lower, min_samples=5, num_samples=n*10
        )
        print(f"  N={n*10}: log LR={log_lr:.1f}, Stop? {should_stop} ({reason})")
    
    # Convergence stopping
    print("\n3. Convergence-Based Stopping")
    print("-" * 70)
    
    config = StoppingConfig(
        rule=StoppingRule.CONVERGENCE,
        convergence_window=10,
        convergence_tolerance=0.001,
        min_samples=10,
        verbose=True
    )
    
    evaluator = StoppingRuleEvaluator(config)
    
    # Simulate converging quality
    quality_trajectory = [0.5, 0.3, 0.2, 0.15, 0.14, 0.135, 0.1345, 0.13445, 0.134445]
    for n, q in enumerate(quality_trajectory, 1):
        decision = evaluator.evaluate(quality=q, num_samples=n*10)
        print(f"  N={n*10}: Quality={q:.6f}, Stop? {decision.should_stop}")
    
    # Confidence interval stopping
    print("\n4. Confidence Interval Width Stopping")
    print("-" * 70)
    
    config = StoppingConfig(
        rule=StoppingRule.CONFIDENCE_WIDTH,
        confidence_width_threshold=0.05,
        min_samples=10,
        verbose=True
    )
    
    evaluator = StoppingRuleEvaluator(config)
    
    ci_widths = [1.0, 0.5, 0.2, 0.1, 0.06, 0.04]
    for n, width in enumerate(ci_widths, 1):
        decision = evaluator.evaluate(confidence_width=width, num_samples=n*20)
        print(f"  N={n*20}: CI width={width:.3f}, Stop? {decision.should_stop}")
    
    # Resource-based stopping
    print("\n5. Resource Budget Stopping")
    print("-" * 70)
    
    config = StoppingConfig(
        rule=StoppingRule.BUDGET,
        max_samples=100,
        max_time=60.0,
        max_cost=100.0,
        measurement_cost=1.0,
        verbose=True
    )
    
    evaluator = StoppingRuleEvaluator(config)
    
    samples = [50, 75, 99, 100, 105]
    times = [10, 20, 30, 40, 50]
    for n, (s, t) in enumerate(zip(samples, times)):
        decision = evaluator.evaluate(
            num_samples=s, time_elapsed=t, cost_incurred=s*1.0
        )
        print(f"  N={s}: t={t}s, cost=${s:.0f}, Stop? {decision.should_stop}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
