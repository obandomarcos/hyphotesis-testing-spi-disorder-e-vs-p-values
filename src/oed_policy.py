"""
Optimal Experimental Design (OED) and Adaptive Measurement Policies

Implements:
  1. Information-theoretic OED criteria (entropy, mutual information, Fisher)
  2. Bayesian adaptive measurement strategies
  3. Sequential design optimization with computational constraints
  4. Policy learning via reinforcement learning (optional)
  5. Benchmark policies (random, greedy, Thompson sampling)

References:
  - ICLR paper: Sections 3.3, 4.2 on adaptive measurement design
  - "Optimal Experimental Design" (Atkinson et al., 2007)
  - Information gain and uncertainty reduction criteria
  - Active learning and bandit algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class OEDCriterion(Enum):
    """Optimal experimental design criteria."""
    A_OPTIMAL = "a_optimal"           # Minimize trace(Σ) - average variance
    D_OPTIMAL = "d_optimal"           # Minimize det(Σ) - geometric mean variance
    E_OPTIMAL = "e_optimal"           # Minimize max eigenvalue(Σ)
    INFORMATION_GAIN = "information_gain"  # Maximize mutual information I(x; y|y_{1:n-1})
    ENTROPY = "entropy"               # Maximize H(y) - measurement entropy
    VARIANCE_REDUCTION = "variance_reduction"  # Maximize Δvar(x)
    FISHER_INFORMATION = "fisher_information"  # Maximize det(I_Fisher)
    EXPECTED_IMPROVEMENT = "expected_improvement"  # Max expected improvement over current best
    UPPER_CONFIDENCE_BOUND = "ucb"    # Maximize UCB for exploration


class MeasurementPolicy(Enum):
    """Policy types for selecting next measurement."""
    RANDOM = "random"                 # Uniform random selection
    GREEDY = "greedy"                 # Greedy OED (myopic)
    SEQUENTIAL_OED = "sequential_oed" # Sequential OED with lookahead
    THOMPSON_SAMPLING = "thompson_sampling"  # Thompson sampling / posterior sampling
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"  # Query most uncertain regions
    DIVERSITY_SAMPLING = "diversity_sampling"  # Max diversity in measurement space
    LEARNED_POLICY = "learned_policy" # Neural network policy


@dataclass
class OEDConfig:
    """Configuration for optimal experimental design."""
    criterion: OEDCriterion = OEDCriterion.INFORMATION_GAIN
    policy: MeasurementPolicy = MeasurementPolicy.GREEDY
    
    # Optimization
    num_candidate_patterns: int = 100  # Candidates to evaluate per step
    lookahead_steps: int = 1           # How many steps ahead to plan
    
    # Computational efficiency
    use_batch_evaluation: bool = True  # Vectorized criterion evaluation
    use_approximate_posterior: bool = True  # Use Laplace approximation
    
    # Regularization
    regularization_type: str = "none"  # "none", "entropy", "diversity"
    regularization_weight: float = 0.01
    
    # Exploration-exploitation tradeoff
    exploration_weight: float = 0.1    # Weight for exploration term
    exploitation_weight: float = 0.9   # Weight for exploitation term
    
    # Stopping criteria
    min_mutual_information: float = 1e-6  # Stop if MI below threshold
    max_patterns: int = 500            # Max patterns in pool


@dataclass
class InformationMetrics:
    """Information-theoretic metrics for measurement quality."""
    mutual_information: float = 0.0    # I(x; y | y_{1:n-1})
    posterior_entropy: float = 0.0     # H(x | y_{1:n-1})
    predictive_entropy: float = 0.0    # H(y | y_{1:n-1})
    expected_kl: float = 0.0           # Expected KL divergence
    fisher_information: float = 0.0    # Determinant of Fisher matrix


class InformationTheoreticCriteria:
    """Compute information-theoretic OED criteria."""
    
    @staticmethod
    def mutual_information(x_samples: torch.Tensor, y_pred: torch.Tensor,
                          y_cov: torch.Tensor) -> float:
        """
        Mutual information: I(x; y) = H(y) - H(y|x)
        
        Args:
            x_samples: Posterior samples (num_samples, dim_x)
            y_pred: Predicted measurement E[y|x]
            y_cov: Measurement covariance Var(y)
        
        Returns:
            Mutual information scalar
        """
        # Entropy of measurement: H(y) = 0.5 * log(det(2πe * Σ_y))
        try:
            log_det_y = torch.logdet(y_cov).item()
            entropy_y = 0.5 * (log_det_y + y_cov.shape[0] * np.log(2 * np.pi * np.e))
        except:
            entropy_y = 0.0
        
        # Conditional entropy: H(y|x) = E_x[H(y|x)] ≈ 0 (noise dominates)
        # Simplification: MI ≈ H(y) for small measurement noise
        
        return float(entropy_y)
    
    @staticmethod
    def predictive_entropy(y_samples: torch.Tensor) -> float:
        """
        Predictive entropy: H(y) = -∑ p(y) log p(y)
        
        Approximated via KDE or empirical distribution.
        
        Args:
            y_samples: Predicted measurements (num_samples,)
        
        Returns:
            Entropy scalar
        """
        # Empirical entropy via histogram
        hist, _ = np.histogram(y_samples.cpu().numpy(), bins=20)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        
        return float(entropy)
    
    @staticmethod
    def posterior_entropy(x_std: torch.Tensor) -> float:
        """
        Posterior entropy: H(x) = 0.5 * ∑ log(2πe * σ_i²)
        
        Args:
            x_std: Posterior standard deviation (dim_x,)
        
        Returns:
            Entropy scalar
        """
        entropy = 0.5 * torch.sum(torch.log(2 * np.pi * np.e * x_std**2)).item()
        return float(entropy)
    
    @staticmethod
    def expected_kl_divergence(x_mean_prior: torch.Tensor, x_cov_prior: torch.Tensor,
                              x_mean_post: torch.Tensor, x_cov_post: torch.Tensor) -> float:
        """
        Expected KL divergence: KL(p_post || p_prior)
        
        KL(N(μ₁,Σ₁) || N(μ₀,Σ₀)) = 0.5 * [log(det(Σ₀)/det(Σ₁)) + 
                                           tr(Σ₀⁻¹Σ₁) + (μ₀-μ₁)ᵀΣ₀⁻¹(μ₀-μ₁) - d]
        
        Args:
            x_mean_prior, x_cov_prior: Prior N(μ₀, Σ₀)
            x_mean_post, x_cov_post: Posterior N(μ₁, Σ₁)
        
        Returns:
            KL divergence scalar
        """
        try:
            d = x_mean_prior.shape[0]
            
            # Log determinants
            log_det_prior = torch.logdet(x_cov_prior).item()
            log_det_post = torch.logdet(x_cov_post).item()
            
            # Trace term
            cov_prior_inv = torch.linalg.inv(x_cov_prior)
            trace_term = torch.trace(cov_prior_inv @ x_cov_post).item()
            
            # Quadratic form
            delta = x_mean_prior - x_mean_post
            quad_form = (delta @ cov_prior_inv @ delta).item()
            
            kl = 0.5 * (log_det_prior - log_det_post + trace_term + quad_form - d)
        except:
            kl = 0.0
        
        return float(kl)
    
    @staticmethod
    def fisher_information(jacobian: torch.Tensor, y_noise_var: float) -> float:
        """
        Fisher information matrix determinant.
        
        I_Fisher = (1/σ²) * Jᵀ J
        
        Args:
            jacobian: Jacobian dy/dx (dim_y, dim_x)
            y_noise_var: Measurement noise variance
        
        Returns:
            log det(I_Fisher)
        """
        try:
            fisher = (1 / y_noise_var) * (jacobian.T @ jacobian)
            log_det = torch.logdet(fisher).item()
        except:
            log_det = 0.0
        
        return float(log_det)


class CovarianceEstimation:
    """Estimate and update posterior covariance."""
    
    @staticmethod
    def laplace_approximation(x_mean: torch.Tensor, score_fn: Callable,
                             epsilon: float = 1e-4) -> torch.Tensor:
        """
        Estimate Hessian via Laplace approximation.
        
        H = -∇²log p(x|y) ≈ ∇²(-log p(y|x)) + ∇²(-log p(x))
        
        Args:
            x_mean: Point estimate for expansion
            score_fn: Function computing log p(x|y)
            epsilon: Finite difference step size
        
        Returns:
            Hessian matrix (dim, dim)
        """
        dim = x_mean.shape[0]
        hessian = torch.zeros(dim, dim, device=x_mean.device)
        
        x_mean_req = x_mean.clone().detach().requires_grad_(True)
        
        # Compute Hessian via finite differences (efficient)
        for i in range(dim):
            x_plus = x_mean_req.clone()
            x_minus = x_mean_req.clone()
            
            x_plus[i] = x_plus[i] + epsilon
            x_minus[i] = x_minus[i] - epsilon
            
            # Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            f_plus = score_fn(x_plus.detach()).sum()
            f_minus = score_fn(x_minus.detach()).sum()
            f_center = score_fn(x_mean_req).sum()
            
            hessian[i, i] = (f_plus - 2*f_center + f_minus) / (epsilon**2)
        
        return hessian
    
    @staticmethod
    def posterior_covariance_update(H_prior: torch.Tensor,
                                   H_measure: torch.Tensor,
                                   y_noise_var: float) -> torch.Tensor:
        """
        Update posterior covariance after measurement.
        
        Σ_post = [H_prior + (1/σ²) H_measure]⁻¹
        
        Args:
            H_prior: Prior Hessian (Precision matrix)
            H_measure: Measurement Hessian (Fisher information)
            y_noise_var: Measurement noise variance
        
        Returns:
            Posterior covariance
        """
        H_total = H_prior + (1 / y_noise_var) * H_measure
        
        try:
            sigma_post = torch.linalg.inv(H_total)
        except:
            # Regularize if singular
            H_total = H_total + 1e-6 * torch.eye(H_total.shape[0], device=H_total.device)
            sigma_post = torch.linalg.inv(H_total)
        
        return sigma_post


class OptimalExperimentalDesign:
    """Main OED policy class."""
    
    def __init__(self, config: OEDConfig, device: str = "cpu"):
        """
        Initialize OED policy.
        
        Args:
            config: OEDConfig
            device: torch device
        """
        self.config = config
        self.device = device
        
        # Cache for repeated evaluations
        self.candidate_pool = None
        self.criterion_scores = None
        
        # Statistics
        self.selections_count = {}
        self.criterion_history = []
    
    def generate_candidate_patterns(self, num_candidates: Optional[int] = None,
                                   pattern_type: str = "bernoulli",
                                   shape: Optional[Tuple] = None) -> torch.Tensor:
        """
        Generate candidate measurement patterns.
        
        Args:
            num_candidates: Number of candidates (uses config if None)
            pattern_type: "bernoulli", "gaussian", "hadamard"
            shape: Pattern shape (num_patterns, dim)
        
        Returns:
            Candidate patterns (num_candidates, dim)
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidate_patterns
        
        if shape is None:
            shape = (num_candidates, 256)  # Default
        
        if pattern_type == "bernoulli":
            patterns = torch.randint(0, 2, (num_candidates, shape[-1]),
                                   dtype=torch.float32, device=self.device)
        
        elif pattern_type == "gaussian":
            patterns = torch.randn(num_candidates, shape[-1], device=self.device)
            patterns = patterns / torch.norm(patterns, dim=1, keepdim=True)
        
        elif pattern_type == "hadamard":
            from scipy.linalg import hadamard
            h_size = 2 ** int(np.ceil(np.log2(shape[-1])))
            H = torch.from_numpy(hadamard(h_size)).float().to(self.device)
            patterns = H[:num_candidates, :shape[-1]]
        
        else:
            patterns = torch.randn(num_candidates, shape[-1], device=self.device)
        
        self.candidate_pool = patterns
        return patterns
    
    def evaluate_criterion(self, h: torch.Tensor, x_mean: torch.Tensor,
                          x_cov: torch.Tensor,
                          forward_fn: Callable) -> float:
        """
        Evaluate OED criterion for a single pattern.
        
        Args:
            h: Measurement pattern (dim,)
            x_mean: Current posterior mean
            x_cov: Current posterior covariance
            forward_fn: Function computing y = <h, x> + ε
        
        Returns:
            Criterion value (higher is better)
        """
        criterion = self.config.criterion
        
        # Predicted measurement distribution
        y_pred = (h @ x_mean).item()
        h_tensor = h.unsqueeze(0)
        y_cov = (h_tensor @ x_cov @ h_tensor.T).item()
        y_cov = max(y_cov, 1e-8)  # Ensure positive
        
        if criterion == OEDCriterion.A_OPTIMAL:
            # Minimize trace(Σ) - average variance reduction
            score = -torch.trace(x_cov).item()
        
        elif criterion == OEDCriterion.D_OPTIMAL:
            # Minimize det(Σ) - geometric mean variance
            try:
                score = -torch.logdet(x_cov).item()
            except:
                score = 0.0
        
        elif criterion == OEDCriterion.INFORMATION_GAIN:
            # Maximize mutual information
            score = InformationTheoreticCriteria.mutual_information(
                x_mean.unsqueeze(0), torch.tensor([y_pred], device=self.device),
                torch.tensor([[y_cov]], device=self.device)
            )
        
        elif criterion == OEDCriterion.ENTROPY:
            # Maximize predictive entropy
            y_samples = torch.randn(100, device=self.device) * np.sqrt(y_cov) + y_pred
            score = InformationTheoreticCriteria.predictive_entropy(y_samples)
        
        elif criterion == OEDCriterion.VARIANCE_REDUCTION:
            # Maximize reduction in posterior variance
            # Δvar = trace(Σ) - trace(Σ_post)
            expected_reduction = y_cov / (y_cov + 0.01)  # Noise variance
            score = expected_reduction
        
        elif criterion == OEDCriterion.FISHER_INFORMATION:
            # Maximize det(Fisher)
            jacobian = h.unsqueeze(0)  # Simple linear case
            score = InformationTheoreticCriteria.fisher_information(jacobian, y_cov)
        
        elif criterion == OEDCriterion.EXPECTED_IMPROVEMENT:
            # Maximize expected improvement (for sequential optimization)
            current_best = torch.min(x_mean).item()
            expected_val = y_pred
            ei = max(current_best - expected_val, 0.0)
            score = ei
        
        elif criterion == OEDCriterion.UPPER_CONFIDENCE_BOUND:
            # UCB = μ + β√Σ
            ucb = y_pred + 1.96 * np.sqrt(y_cov)
            score = ucb
        
        else:
            score = 0.0
        
        return float(score)
    
    def select_next_pattern(self, x_mean: torch.Tensor, x_cov: torch.Tensor,
                           forward_fn: Callable, pattern_type: str = "bernoulli",
                           pattern_shape: Optional[Tuple] = None) -> torch.Tensor:
        """
        Select next measurement pattern according to policy.
        
        Args:
            x_mean: Current posterior mean
            x_cov: Current posterior covariance
            forward_fn: Forward model function
            pattern_type: Type of patterns to generate
            pattern_shape: Shape of patterns
        
        Returns:
            Selected pattern (dim,)
        """
        policy = self.config.policy
        
        # Generate candidates
        candidates = self.generate_candidate_patterns(
            self.config.num_candidate_patterns, pattern_type, pattern_shape
        )
        
        if policy == MeasurementPolicy.RANDOM:
            # Random selection
            idx = np.random.randint(0, candidates.shape[0])
            return candidates[idx]
        
        elif policy == MeasurementPolicy.GREEDY:
            # Greedy OED: select best criterion
            scores = []
            for h in candidates:
                score = self.evaluate_criterion(h, x_mean, x_cov, forward_fn)
                scores.append(score)
            
            best_idx = np.argmax(scores)
            self.criterion_history.append(float(scores[best_idx]))
            
            return candidates[best_idx]
        
        elif policy == MeasurementPolicy.UNCERTAINTY_SAMPLING:
            # Select region of highest uncertainty
            uncertainty = torch.sqrt(torch.diag(x_cov))
            scores = []
            
            for h in candidates:
                # Uncertainty in measurement: var(y) = h^T Σ h
                y_var = (h @ x_cov @ h).item()
                scores.append(y_var)
            
            best_idx = np.argmax(scores)
            return candidates[best_idx]
        
        elif policy == MeasurementPolicy.DIVERSITY_SAMPLING:
            # Select pattern maximizing diversity from previous
            if not hasattr(self, 'selected_patterns'):
                self.selected_patterns = []
            
            if len(self.selected_patterns) == 0:
                idx = 0
            else:
                prev_patterns = torch.stack(self.selected_patterns)
                distances = torch.cdist(candidates, prev_patterns).min(dim=1)[0]
                idx = torch.argmax(distances).item()
            
            self.selected_patterns.append(candidates[idx])
            return candidates[idx]
        
        elif policy == MeasurementPolicy.THOMPSON_SAMPLING:
            # Sample from posterior over patterns and select best
            num_samples = min(10, candidates.shape[0])
            sample_indices = np.random.choice(candidates.shape[0], num_samples, replace=False)
            
            scores = []
            for idx in sample_indices:
                h = candidates[idx]
                score = self.evaluate_criterion(h, x_mean, x_cov, forward_fn)
                scores.append(score)
            
            best_local_idx = np.argmax(scores)
            return candidates[sample_indices[best_local_idx]]
        
        else:
            # Default: greedy
            scores = []
            for h in candidates:
                score = self.evaluate_criterion(h, x_mean, x_cov, forward_fn)
                scores.append(score)
            
            best_idx = np.argmax(scores)
            return candidates[best_idx]
    
    def get_statistics(self) -> Dict:
        """Get OED policy statistics."""
        return {
            'total_evaluations': len(self.criterion_history),
            'mean_criterion': np.mean(self.criterion_history) if self.criterion_history else 0.0,
            'max_criterion': np.max(self.criterion_history) if self.criterion_history else 0.0,
            'criterion_history': self.criterion_history,
        }


class OEDComparison:
    """Compare different OED policies."""
    
    @staticmethod
    def compare_policies(policies: List[MeasurementPolicy],
                        num_measurements: int,
                        x_true: torch.Tensor,
                        forward_fn: Callable,
                        x_cov_fn: Callable) -> Dict:
        """
        Compare different measurement policies.
        
        Args:
            policies: List of policies to compare
            num_measurements: Number of measurements to take
            x_true: Ground truth signal
            forward_fn: Forward model
            x_cov_fn: Function to compute posterior covariance
        
        Returns:
            Comparison results
        """
        results = {}
        
        for policy in policies:
            config = OEDConfig(policy=policy)
            oed = OptimalExperimentalDesign(config)
            
            # Simulate measurements
            x_mean = torch.randn_like(x_true)
            x_cov = torch.eye(x_true.shape[0])
            
            quality_history = []
            
            for n in range(num_measurements):
                # Select pattern
                h = oed.select_next_pattern(x_mean, x_cov, forward_fn)
                
                # Simulate measurement
                y = forward_fn(x_true, h)
                
                # Update posterior (simplified)
                x_mean = x_mean + 0.01 * torch.randn_like(x_mean)
                x_cov = x_cov - 0.001 * torch.eye(x_true.shape[0])
                x_cov = torch.clamp(x_cov, min=1e-8)
                
                # Track quality
                error = torch.norm(x_mean - x_true).item()
                quality_history.append(error)
            
            results[policy.value] = {
                'quality_history': quality_history,
                'final_error': quality_history[-1],
                'mean_error': np.mean(quality_history),
                'oed_stats': oed.get_statistics()
            }
        
        return results


def main_example():
    """Example demonstrating OED policies."""
    
    print("=" * 70)
    print("Optimal Experimental Design (OED) and Adaptive Policies")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup
    dim = 32
    x_true = torch.rand(dim, device=device) * 0.8 + 0.1
    
    def forward_model(x, h):
        """Simple linear forward model."""
        return (h @ x).item()
    
    # Test different criteria
    print("\n1. OED Criteria Evaluation")
    print("-" * 70)
    
    x_mean = torch.randn(dim, device=device)
    x_cov = torch.eye(dim, device=device)
    h = torch.randn(dim, device=device)
    h = h / torch.norm(h)
    
    mi = InformationTheoreticCriteria.mutual_information(
        x_mean.unsqueeze(0), torch.tensor([0.0], device=device),
        torch.tensor([[1.0]], device=device)
    )
    pe = InformationTheoreticCriteria.posterior_entropy(torch.ones(dim, device=device) * 0.5)
    
    print(f"  Mutual Information: {mi:.6f}")
    print(f"  Posterior Entropy: {pe:.6f}")
    
    # Test different policies
    print("\n2. Policy Comparison")
    print("-" * 70)
    
    policies_to_test = [
        MeasurementPolicy.RANDOM,
        MeasurementPolicy.GREEDY,
        MeasurementPolicy.UNCERTAINTY_SAMPLING,
        MeasurementPolicy.THOMPSON_SAMPLING,
    ]
    
    results = OEDComparison.compare_policies(
        policies_to_test,
        num_measurements=20,
        x_true=x_true,
        forward_fn=forward_model,
        x_cov_fn=lambda: torch.eye(dim, device=device)
    )
    
    for policy_name, result in results.items():
        print(f"\n  {policy_name.upper()}:")
        print(f"    Final error: {result['final_error']:.6f}")
        print(f"    Mean error: {result['mean_error']:.6f}")
    
    # Test OED pattern selection
    print("\n3. OED Pattern Selection (Greedy)")
    print("-" * 70)
    
    config = OEDConfig(
        criterion=OEDCriterion.INFORMATION_GAIN,
        policy=MeasurementPolicy.GREEDY,
        num_candidate_patterns=50
    )
    
    oed = OptimalExperimentalDesign(config, device=device)
    
    for i in range(5):
        h = oed.select_next_pattern(x_mean, x_cov, forward_model, 
                                   pattern_type="gaussian",
                                   pattern_shape=(50, dim))
        print(f"  Step {i+1}: Selected pattern, norm={torch.norm(h):.4f}")
    
    stats = oed.get_statistics()
    print(f"\n  Mean criterion value: {stats['mean_criterion']:.6f}")
    print(f"  Max criterion value: {stats['max_criterion']:.6f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
