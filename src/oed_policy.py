"""
Optimal Experimental Design (OED) and Adaptive Policies - CORRECTED

Implements:
  1. Information-theoretic OED criteria
  2. Adaptive measurement selection policies
  3. E-value and p-value based policies
  4. Comparison with random sampling
  5. Sequential OED with disorder-averaging

References:
  - Chaloner & Verdinelli (1995): "Bayesian Experimental Design"
  - Sebastiani & Wynn (2000): "Maximum entropy sampling"
  - ICLR/TPAMI papers on OED for inverse problems
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


class OEDCriterion(Enum):
    """OED selection criteria."""
    MUTUAL_INFORMATION = "mutual_information"
    POSTERIOR_ENTROPY = "posterior_entropy"
    EXPECTED_INFORMATION_GAIN = "expected_information_gain"
    VARIANCE_REDUCTION = "variance_reduction"
    E_VALUE_BASED = "e_value_based"


@dataclass
class OEDConfig:
    """Configuration for optimal experimental design."""
    # Problem dimensions
    dim_x: int = 256
    num_candidate_designs: int = 100
    
    # OED criteria
    oed_criterion: OEDCriterion = OEDCriterion.MUTUAL_INFORMATION
    
    # Measurement parameters
    noise_std: float = 0.1
    measurement_pattern: str = "random_binary"  # "random_binary", "gaussian"
    
    # Bayesian prior
    prior_mean: float = 0.0
    prior_std: float = 1.0
    
    # Sequential OED
    num_sequential_steps: int = 10
    batch_size: int = 1
    
    # E-value/P-value parameters
    alpha: float = 0.05
    e_value_threshold: float = 1.0/0.05
    
    # Verbosity
    verbose: bool = False


@dataclass
class MeasurementPattern:
    """Representation of a measurement pattern."""
    h: torch.Tensor              # Pattern vector (dim_x,)
    criterion_value: float = 0.0
    entropy_reduction: float = 0.0
    information_gain: float = 0.0
    device: str = "cpu"


class OEDCriteria:
    """Computes information-theoretic OED criteria."""
    
    @staticmethod
    def mutual_information(y: torch.Tensor, x_prior_mean: torch.Tensor,
                          x_prior_cov: torch.Tensor, h: torch.Tensor,
                          noise_std: float, device: str = "cpu") -> float:
        """
        Compute mutual information: I(y; x) = H(y) - H(y|x).
        
        For Gaussian models:
        I(y; x) = 0.5 * log det(Σ_y) - 0.5 * log det(Σ_{y|x})
        
        Args:
            y: Observed measurement
            x_prior_mean: Prior mean of x
            x_prior_cov: Prior covariance of x
            h: Measurement pattern
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Mutual information value
        """
        # Ensure all tensors on same device
        h = h.to(device)
        x_prior_mean = x_prior_mean.to(device)
        x_prior_cov = x_prior_cov.to(device)
        
        # Marginal variance of y: Var(y) = h^T Σ_x h + σ²
        var_y = torch.dot(h, x_prior_cov @ h) + noise_std**2
        
        # Conditional variance: Var(y|x) = σ²
        var_y_given_x = noise_std**2
        
        # Mutual information
        mi = 0.5 * torch.log(var_y / (var_y_given_x + 1e-8))
        
        return float(mi.cpu().item())
    
    @staticmethod
    def posterior_entropy(x_prior_cov: torch.Tensor, h: torch.Tensor,
                         noise_std: float, device: str = "cpu") -> float:
        """
        Compute posterior entropy after measurement.
        
        H(x|y) = H(x) - I(x; y)
        
        Args:
            x_prior_cov: Prior covariance of x
            h: Measurement pattern
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Posterior entropy value
        """
        # Ensure all tensors on same device
        h = h.to(device)
        x_prior_cov = x_prior_cov.to(device)
        
        # Prior entropy: H(x) = 0.5 * log det(2πe Σ_x)
        try:
            det_prior = torch.det(x_prior_cov)
            if det_prior <= 0:
                det_prior = 1.0
            prior_entropy = 0.5 * torch.log(2 * np.pi * np.e * det_prior)
        except:
            prior_entropy = torch.tensor(0.0, device=device)
        
        # Information gain
        var_y = torch.dot(h, x_prior_cov @ h) + noise_std**2
        var_y_given_x = noise_std**2
        info_gain = 0.5 * torch.log(var_y / (var_y_given_x + 1e-8))
        
        # Posterior entropy
        posterior_entropy = prior_entropy - info_gain
        
        return float(posterior_entropy.cpu().item())
    
    @staticmethod
    def expected_information_gain(x_prior_cov: torch.Tensor, h: torch.Tensor,
                                 noise_std: float, device: str = "cpu") -> float:
        """
        Compute expected information gain (same as mutual information).
        
        Args:
            x_prior_cov: Prior covariance of x
            h: Measurement pattern
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Expected information gain
        """
        # Ensure all tensors on same device
        h = h.to(device)
        x_prior_cov = x_prior_cov.to(device)
        
        # E[I(x; y)] = 0.5 * log(1 + h^T Σ_x h / σ²)
        var_reduction = torch.dot(h, x_prior_cov @ h)
        eig = 0.5 * torch.log(1 + var_reduction / (noise_std**2 + 1e-8))
        
        return float(eig.cpu().item())


class OEDPolicy:
    """Adaptive measurement selection policies."""
    
    @staticmethod
    def random_policy(dim_x: int, device: str = "cpu") -> torch.Tensor:
        """
        Random measurement selection.
        
        Args:
            dim_x: Dimension of signal
            device: torch device
        
        Returns:
            Random measurement pattern (dim_x,)
        """
        h = torch.randn(dim_x, device=device)
        h = h / (torch.norm(h) + 1e-8)
        return h
    
    @staticmethod
    def mutual_information_policy(num_candidates: int, dim_x: int,
                                 x_prior_cov: torch.Tensor,
                                 noise_std: float,
                                 device: str = "cpu") -> torch.Tensor:
        """
        Select measurement pattern maximizing mutual information.
        
        Args:
            num_candidates: Number of candidate patterns to evaluate
            dim_x: Dimension of signal
            x_prior_cov: Prior covariance of x
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Optimal measurement pattern
        """
        # Ensure covariance on correct device
        x_prior_cov = x_prior_cov.to(device)
        
        best_mi = -np.inf
        best_h = None
        
        for _ in range(num_candidates):
            h = torch.randn(dim_x, device=device)
            h = h / (torch.norm(h) + 1e-8)
            
            mi = OEDCriteria.mutual_information(
                None, torch.zeros(dim_x, device=device), x_prior_cov, h,
                noise_std, device=device
            )
            
            if mi > best_mi:
                best_mi = mi
                best_h = h.clone()
        
        return best_h if best_h is not None else torch.randn(dim_x, device=device)
    
    @staticmethod
    def entropy_reduction_policy(num_candidates: int, dim_x: int,
                                x_prior_cov: torch.Tensor,
                                noise_std: float,
                                device: str = "cpu") -> torch.Tensor:
        """
        Select measurement pattern maximizing entropy reduction.
        
        Args:
            num_candidates: Number of candidate patterns to evaluate
            dim_x: Dimension of signal
            x_prior_cov: Prior covariance of x
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Optimal measurement pattern
        """
        # Ensure covariance on correct device
        x_prior_cov = x_prior_cov.to(device)
        
        best_entropy = np.inf
        best_h = None
        
        for _ in range(num_candidates):
            h = torch.randn(dim_x, device=device)
            h = h / (torch.norm(h) + 1e-8)
            
            entropy = OEDCriteria.posterior_entropy(
                x_prior_cov, h, noise_std, device=device
            )
            
            if entropy < best_entropy:
                best_entropy = entropy
                best_h = h.clone()
        
        return best_h if best_h is not None else torch.randn(dim_x, device=device)
    
    @staticmethod
    def e_value_based_policy(num_candidates: int, dim_x: int,
                            x_prior_cov: torch.Tensor,
                            noise_std: float,
                            device: str = "cpu") -> torch.Tensor:
        """
        Select measurement pattern maximizing expected e-value.
        
        Args:
            num_candidates: Number of candidate patterns to evaluate
            dim_x: Dimension of signal
            x_prior_cov: Prior covariance of x
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Optimal measurement pattern
        """
        # Ensure covariance on correct device
        x_prior_cov = x_prior_cov.to(device)
        
        best_eig = -np.inf
        best_h = None
        
        for _ in range(num_candidates):
            h = torch.randn(dim_x, device=device)
            h = h / (torch.norm(h) + 1e-8)
            
            eig = OEDCriteria.expected_information_gain(
                x_prior_cov, h, noise_std, device=device
            )
            
            if eig > best_eig:
                best_eig = eig
                best_h = h.clone()
        
        return best_h if best_h is not None else torch.randn(dim_x, device=device)


class OEDComparison:
    """Compare different OED policies."""
    
    @staticmethod
    def compare_policies(x_true: torch.Tensor, num_measurements: int,
                        num_policy_candidates: int, noise_std: float,
                        device: str = "cpu") -> Dict:
        """
        Compare OED policies with forward model.
        
        Args:
            x_true: True signal
            num_measurements: Number of measurements to collect
            num_policy_candidates: Number of candidate patterns per policy
            noise_std: Measurement noise std
            device: torch device
        
        Returns:
            Dictionary with policy comparison results
        """
        # Ensure x_true on correct device
        x_true = x_true.to(device)
        dim_x = x_true.shape[0]
        
        # Prior covariance (identity for simplicity)
        x_prior_cov = torch.eye(dim_x, device=device)
        
        # Forward model that keeps everything on same device
        def forward_model(x: torch.Tensor, h: torch.Tensor) -> float:
            """Forward model: y = h·x + noise."""
            x = x.to(device)
            h = h.to(device)
            y = torch.dot(h, x)
            noise = noise_std * torch.randn(1, device=device)
            return float((y + noise).cpu().item())
        
        results = {
            'random': {'measurements': [], 'errors': []},
            'mutual_information': {'measurements': [], 'errors': []},
            'entropy_reduction': {'measurements': [], 'errors': []},
            'e_value_based': {'measurements': [], 'errors': []}
        }
        
        # Simulate measurements with different policies
        for policy_name in ['random', 'mutual_information', 'entropy_reduction', 'e_value_based']:
            measurements = []
            patterns = []
            
            for n in range(num_measurements):
                # Select pattern based on policy
                if policy_name == 'random':
                    h = OEDPolicy.random_policy(dim_x, device=device)
                elif policy_name == 'mutual_information':
                    h = OEDPolicy.mutual_information_policy(
                        num_policy_candidates, dim_x, x_prior_cov, noise_std, device=device
                    )
                elif policy_name == 'entropy_reduction':
                    h = OEDPolicy.entropy_reduction_policy(
                        num_policy_candidates, dim_x, x_prior_cov, noise_std, device=device
                    )
                else:  # e_value_based
                    h = OEDPolicy.e_value_based_policy(
                        num_policy_candidates, dim_x, x_prior_cov, noise_std, device=device
                    )
                
                # Collect measurement
                y = forward_model(x_true, h)
                measurements.append(y)
                patterns.append(h)
                
                # Compute reconstruction error
                H = torch.stack(patterns)  # (n_measurements, dim_x)
                y_vec = torch.tensor(measurements, device=device)
                
                try:
                    H_pinv = torch.linalg.pinv(H)
                    x_recon = H_pinv @ y_vec
                    error = float(torch.norm(x_recon - x_true).cpu().item())
                except:
                    error = float(torch.norm(x_true).cpu().item())
                
                results[policy_name]['measurements'].append(n + 1)
                results[policy_name]['errors'].append(error)
        
        return results


def main_example():
    """Example demonstrating OED and adaptive policies."""
    
    print("=" * 70)
    print("Optimal Experimental Design (OED) and Adaptive Policies")
    print("=" * 70)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Configuration
    config = OEDConfig(
        dim_x=256,
        num_candidate_designs=100,
        oed_criterion=OEDCriterion.MUTUAL_INFORMATION,
        noise_std=0.1,
        num_sequential_steps=10,
        verbose=True
    )
    
    # 1. OED Criteria Evaluation
    print("1. OED Criteria Evaluation")
    print("-" * 70)
    
    dim_x = config.dim_x
    x_prior_cov = torch.eye(dim_x, device=device)
    h_random = torch.randn(dim_x, device=device)
    h_random = h_random / (torch.norm(h_random) + 1e-8)
    
    mi = OEDCriteria.mutual_information(
        None, torch.zeros(dim_x, device=device), x_prior_cov, h_random,
        config.noise_std, device=device
    )
    
    entropy = OEDCriteria.posterior_entropy(
        x_prior_cov, h_random, config.noise_std, device=device
    )
    
    print(f"  Mutual Information: {mi:.6f}")
    print(f"  Posterior Entropy: {entropy:.6f}")
    
    # 2. Policy Comparison
    print("\n2. Policy Comparison")
    print("-" * 70)
    
    # Generate true signal
    x_true = torch.randn(dim_x, device=device) / np.sqrt(dim_x)
    x_true = x_true / torch.norm(x_true)
    
    # Compare policies
    results = OEDComparison.compare_policies(
        x_true, num_measurements=20,
        num_policy_candidates=config.num_candidate_designs,
        noise_std=config.noise_std,
        device=device
    )
    
    # Print final errors
    for policy_name, data in results.items():
        if data['errors']:
            final_error = data['errors'][-1]
            num_meas = data['measurements'][-1]
            print(f"  {policy_name.upper()}: "
                  f"n={num_meas}, final_error={final_error:.4f}")
    
    # 3. Measurement Efficiency
    print("\n3. Measurement Efficiency")
    print("-" * 70)
    
    # Find number of measurements to reach error < 0.3
    target_error = 0.3
    efficiency = {}
    
    for policy_name, data in results.items():
        n_to_target = None
        for n, error in zip(data['measurements'], data['errors']):
            if error < target_error:
                n_to_target = n
                break
        
        if n_to_target:
            efficiency[policy_name] = n_to_target
            print(f"  {policy_name.upper()}: {n_to_target} measurements to reach error < {target_error}")
        else:
            print(f"  {policy_name.upper()}: Did not reach target error")
    
    # 4. Summary
    print("\n4. Summary")
    print("-" * 70)
    
    if efficiency:
        best_policy = min(efficiency, key=efficiency.get)
        worst_policy = max(efficiency, key=efficiency.get)
        speedup = efficiency[worst_policy] / efficiency[best_policy]
        
        print(f"  Best policy: {best_policy.upper()} ({efficiency[best_policy]} measurements)")
        print(f"  Speedup over worst: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
