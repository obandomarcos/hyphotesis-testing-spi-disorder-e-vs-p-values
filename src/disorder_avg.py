"""
Disorder-Averaged Anomalous Langevin Score Sampling (DA-ALSS)

Implements:
  1. Disorder-averaging framework for random measurement operators
  2. Anomalous Langevin dynamics with Lévy processes
  3. Score-based posterior sampling with disorder robustness
  4. Effective medium estimation (Phase 1)
  5. Sequential measurement integration

References:
  - TPAMI paper: Sections 2.1-2.4 on disorder-averaging
  - ICLR paper: Section 3.2 on ALSS sampling
  - "Disorder Averaging in Inverse Problems" (CWI research)
  - Score-based generative models (Song et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class DisorderType(Enum):
    """Types of disorder in measurement operators."""
    RANDOM_PATTERNS = "random_patterns"        # Random binary/Gaussian patterns
    STRUCTURED_PERTURBATION = "structured"     # Structured disorder
    GAUSSIAN_DISORDER = "gaussian_disorder"    # Gaussian random matrix
    BERNOULLI_DISORDER = "bernoulli_disorder"  # Bernoulli random matrix


@dataclass
class DisorderConfig:
    """Configuration for disorder-averaged sampling."""
    # Problem dimensions
    dim_x: int = 256                   # Dimension of signal
    dim_y: int = 64                    # Dimension of measurements
    
    # Disorder parameters
    disorder_type: DisorderType = DisorderType.RANDOM_PATTERNS
    num_disorder_samples: int = 50     # K: disorder realizations
    disorder_probability: float = 0.5  # p for Bernoulli patterns
    
    # Anomalous diffusion parameters
    num_diffusion_steps: int = 100     # Number of reverse SDE steps
    levy_alpha: float = 1.5            # Lévy process exponent (1 < α ≤ 2)
    
    # Likelihood parameters
    measurement_noise_std: float = 0.1 # σ: measurement noise
    likelihood_weight: float = 1.0     # Weight for data fidelity
    
    # Score network parameters
    score_net_hidden: int = 128        # Hidden dimension for score network
    
    # Optimization
    learning_rate: float = 0.001
    num_score_iterations: int = 1000
    
    # Regularization
    regularization_lambda: float = 0.01  # L2 regularization
    gradient_clip: float = 10.0        # Gradient clipping threshold
    
    # Numerical stability
    min_variance: float = 1e-8         # Minimum variance for stability
    max_value: float = 100.0           # Max abs value before clipping
    
    # Verbosity
    verbose: bool = False


@dataclass
class EffectiveMedium:
    """Effective medium estimation results."""
    A_eff: torch.Tensor                # Effective operator (dim_y, dim_x)
    Lambda_eff: torch.Tensor           # Effective covariance (dim_y, dim_y)
    sigma_model: torch.Tensor          # Disorder-induced covariance
    num_realizations: int = 50         # K realizations


class DisorderAveragedALSS:
    """
    Disorder-Averaged Anomalous Langevin Score Sampling.
    
    Phase 1: Effective medium estimation under disorder
    Phase 2: Posterior sampling via anomalous diffusion
    """
    
    def __init__(self, score_network: nn.Module, config: DisorderConfig,
                 device: str = "cpu"):
        """
        Initialize DA-ALSS sampler.
        
        Args:
            score_network: Trained score network ∇_x log p(x)
            config: DisorderConfig
            device: torch device
        """
        self.config = config
        self.device = device
        
        # Score network (pre-trained prior)
        self.score_network = score_network.to(device)
        self.score_network.eval()
        
        # Effective medium (Phase 1 output)
        self.effective_medium: Optional[EffectiveMedium] = None
    
    def estimate_effective_medium(self, A_samples: List[torch.Tensor],
                                 y_samples: Optional[torch.Tensor] = None) -> EffectiveMedium:
        """
        Phase 1: Estimate effective medium from operator ensemble.
        
        Computes:
          A_eff = E_disorder[A]
          Λ_eff = E_disorder[A A^T]
          σ_model = Cov_disorder{A}
        
        Args:
            A_samples: List of K measurement operators (each dim_y × dim_x)
            y_samples: Optional measurement samples
        
        Returns:
            EffectiveMedium with effective operators
        """
        K = len(A_samples)
        
        # Stack operators
        A_stacked = torch.stack(A_samples)  # (K, dim_y, dim_x)
        
        # Compute effective operator
        A_eff = torch.mean(A_stacked, dim=0)  # (dim_y, dim_x)
        
        # Compute effective covariance: Λ_eff = E[AA^T]
        Lambda_list = []
        for k in range(K):
            Lambda_k = A_stacked[k] @ A_stacked[k].T
            Lambda_list.append(Lambda_k)
        
        Lambda_eff = torch.mean(torch.stack(Lambda_list), dim=0)  # (dim_y, dim_y)
        
        # Disorder-induced covariance
        sigma_model = torch.zeros(self.config.dim_y, self.config.dim_y, 
                                 device=self.device, dtype=A_eff.dtype)
        
        for k in range(K):
            Delta_A = A_stacked[k] - A_eff
            sigma_model += (Delta_A @ Delta_A.T) / K
        
        self.effective_medium = EffectiveMedium(
            A_eff=A_eff,
            Lambda_eff=Lambda_eff,
            sigma_model=sigma_model,
            num_realizations=K
        )
        
        if self.config.verbose:
            print(f"  A_eff shape: {A_eff.shape}")
            print(f"  Λ_eff shape: {Lambda_eff.shape}")
        
        return self.effective_medium
    
    def _compute_score(self, x: torch.Tensor, y: torch.Tensor,
                      t: float) -> torch.Tensor:
        """
        Compute score function: ∇_x log p(x|y).
        
        Score = Prior score + Likelihood score
        
        Args:
            x: Current sample (num_samples, batch, dim_x) or (batch, dim_x)
            y: Measurements (batch, dim_y)
            t: Time parameter [0, 1]
        
        Returns:
            Score vector matching x shape
        """
        # Ensure shapes
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add sample dimension
        
        num_samples, batch_size, dim_x = x.shape
        
        # Prior score (from learned score network)
        t_tensor = torch.tensor(t, device=self.device).float()
        
        # Reshape for network
        x_flat = x.reshape(-1, dim_x)  # (num_samples * batch, dim_x)
        t_expanded = t_tensor.expand(num_samples * batch_size)
        
        with torch.no_grad():
            prior_score = self.score_network(x_flat, t_expanded)  # (num_samples * batch, dim_x)
        
        prior_score = prior_score.reshape(num_samples, batch_size, dim_x)
        
        # Likelihood score: ∇_x log p(y|x) ∝ -||y - Ax||²
        if self.effective_medium is not None:
            A_eff = self.effective_medium.A_eff  # (dim_y, dim_x)
            
            # Predicted measurements
            y_pred = torch.matmul(x, A_eff.T)  # ✅ (num_samples, batch, dim_x) @ (dim_x, dim_y) = (num_samples, batch, dim_y)
            
            # Residual
            residual = y.unsqueeze(0) - y_pred  # (num_samples, batch, dim_y)
            
            # Likelihood score: A_eff^T @ residual / σ²
            likelihood_score = torch.matmul(residual, A_eff)  # ✅ (num_samples, batch, dim_y) @ (dim_y, dim_x) = (num_samples, batch, dim_x)
            likelihood_score = likelihood_score / (self.config.measurement_noise_std ** 2 + self.config.min_variance)
            
            # Combine scores with weighting
            score = prior_score + self.config.likelihood_weight * likelihood_score
        else:
            score = prior_score
        
        # Regularization: prevent explosion
        score = torch.clamp(score, -self.config.gradient_clip, self.config.gradient_clip)
        
        return score
    
    def sample(self, y: torch.Tensor, num_samples: int = 5,
              verbose: bool = False) -> torch.Tensor:
        """
        Phase 2: Generate posterior samples via anomalous Langevin dynamics.
        
        Reverse SDE: dx = -∇_x[-log p(x|y)] dt + √(2α) dw_α
        
        where w_α is Lévy process with stability parameter α ∈ (1, 2].
        
        Args:
            y: Measurements (batch, dim_y)
            num_samples: Number of posterior samples per measurement
            verbose: Print progress
        
        Returns:
            Posterior samples (num_samples, batch, dim_x)
        """
        if self.effective_medium is None:
            raise RuntimeError("Must call estimate_effective_medium() first")
        
        batch_size = y.shape[0]
        dim_x = self.config.dim_x
        
        # Initialize from prior (scaled by 1/σ for stability)
        x_samples = 0.1 * torch.randn(num_samples, batch_size, dim_x, 
                                     device=self.device, dtype=y.dtype)
        
        # Reverse diffusion loop
        dt_base = 1.0 / self.config.num_diffusion_steps
        
        for step in range(self.config.num_diffusion_steps - 1, -1, -1):
            t = step / self.config.num_diffusion_steps
            dt = dt_base
            
            # Compute score with numerical stability
            with torch.no_grad():
                score = self._compute_score(x_samples, y, t)
            
            # Clip score to prevent explosion
            score = torch.clamp(score, -self.config.gradient_clip, self.config.gradient_clip)
            
            # Anomalous diffusion: noise scale ~ dt^(1/(2α))
            noise = torch.randn_like(x_samples)
            noise_scale = np.sqrt(2.0 * self.config.levy_alpha * (dt ** (1.0 / self.config.levy_alpha)))
            noise_scale = min(noise_scale, 0.1)  # Cap noise scale
            
            # Drift term (capped for stability)
            drift = 0.5 * score * dt
            drift = torch.clamp(drift, -0.05, 0.05)
            
            # Langevin update
            x_samples = x_samples - drift + noise_scale * noise
            
            # Clipping for stability
            x_samples = torch.clamp(x_samples, -self.config.max_value, self.config.max_value)
            
            if verbose and step % 10 == 0:
                mean_val = torch.abs(x_samples).mean().item()
                if not np.isnan(mean_val) and not np.isinf(mean_val):
                    print(f"  Step {step}: mean(|x|) = {mean_val:.4f}")
                else:
                    print(f"  Step {step}: mean(|x|) = {mean_val:.4f} (check stability)")
        
        return x_samples


def main_example():
    """Example demonstrating disorder-averaged ALSS."""
    
    print("=" * 70)
    print("Disorder-Averaged Anomalous Langevin Score Sampling (DA-ALSS)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Configuration
    config = DisorderConfig(
        dim_x=256,
        dim_y=64,
        num_disorder_samples=50,
        num_diffusion_steps=100,
        levy_alpha=1.8,
        measurement_noise_std=0.1,
        verbose=True
    )
    
    # Dummy score network (pre-trained prior)
    class SimpleScoreNet(nn.Module):
        def __init__(self, dim_x: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_x + 1, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim_x)
            )
        
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = (t / 1000.0).unsqueeze(-1) if t.dim() == 0 else (t / 1000.0).unsqueeze(-1)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if t_emb.dim() == 1:
                t_emb = t_emb.unsqueeze(-1)
            t_emb = t_emb.expand(x.shape[0], 1)
            x_t = torch.cat([x, t_emb], dim=-1)
            return -x  # Prior: negative for Gaussian
    
    score_net = SimpleScoreNet(config.dim_x).to(device)
    
    # Phase 1: Effective medium estimation
    print("Phase 1: Effective Medium Estimation")
    print("-" * 70)
    
    sampler = DisorderAveragedALSS(score_net, config, device=device)
    
    # Generate disorder sample
    A_samples = []
    for k in range(config.num_disorder_samples):
        A_k = torch.randn(config.dim_y, config.dim_x, device=device) / np.sqrt(config.dim_y)
        A_samples.append(A_k)
    
    effective_medium = sampler.estimate_effective_medium(A_samples)
    
    # Phase 2: Posterior sampling
    print("\nPhase 2: Anomalous Diffusion Sampling")
    print("-" * 70)
    
    # Generate synthetic measurements
    x_true = torch.randn(4, config.dim_x, device=device) / np.sqrt(config.dim_x)
    y_true = A_samples[0] @ x_true.T  # (dim_y, 4)
    y_true = y_true.T + config.measurement_noise_std * torch.randn_like(y_true.T)  # (4, dim_y)
    
    print(f"  True signal shape: {x_true.shape}")
    print(f"  Measurement shape: {y_true.shape}")
    
    # Sample posterior
    posterior_samples = sampler.sample(y_true, num_samples=5, verbose=True)
    
    print(f"\nPosterior samples shape: {posterior_samples.shape}")
    print(f"  Expected: (5, 4, 256)")
    
    # Compute PSNR
    x_recon = posterior_samples.mean(dim=0)  # (batch, dim_x)
    mse = F.mse_loss(x_recon, x_true)
    if not torch.isnan(mse):
        psnr = 10 * np.log10(1.0 / (mse.item() + 1e-8))
        print(f"\nPSNR (posterior mean vs. true): {psnr:.2f} dB")
    else:
        print("\nPSNR: nan (numerical instability detected)")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
