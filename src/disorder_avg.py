"""
Disorder-Averaged Anomalous Langevin Score Sampling (DA-ALSS)

Implements disorder-aware generative modeling for inverse problems using:
  1. Score-based diffusion models as learned priors
  2. Lévy-stable noise for anomalous diffusion
  3. Disorder-averaging framework for random operators
  4. Effective medium theory for operator uncertainty

References:
  - Anomalous Langevin Score Sampling (ALSS) for disordered systems
  - Disorder-Adaptive diffusion posterior sampling (DA-DPS)
  - Single-pixel imaging with e-value hypothesis testing (ICLR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Callable, List, Union
from dataclasses import dataclass
from scipy.special import gamma as scipy_gamma
import warnings


@dataclass
class DisorderConfig:
    """Configuration for disorder-averaged sampling."""
    # Diffusion process
    num_steps: int = 100
    beta_schedule: str = "linear"  # "linear", "cosine", "sqrt"
    
    # Lévy flight parameters
    levy_alpha: float = 1.5  # Lévy index ∈ (1, 2], smaller = heavier tails
    levy_gamma: float = 1.0  # Scale parameter
    
    # Disorder averaging
    use_disorder_averaging: bool = True
    num_disorder_samples: int = 50  # K in Algorithm 2
    disorder_sigma: float = 0.05  # σ in disorder perturbations
    
    # Likelihood weighting
    likelihood_weight: float = 1.0
    consistency_reg_weight: float = 0.1
    
    # Measurement noise
    measurement_noise_std: float = 1.0
    
    # Optimization
    use_tempered_levy: bool = False  # Truncate large Lévy jumps
    levy_truncation_threshold: float = 10.0


class BetaScheduler:
    """DDPM-style noise schedule."""
    
    def __init__(self, num_steps: int, schedule: str = "linear"):
        self.num_steps = num_steps
        self.schedule = schedule
        self._build_schedule()
    
    def _build_schedule(self):
        """Build noise variance schedule."""
        if self.schedule == "linear":
            self.betas = torch.linspace(0.0001, 0.02, self.num_steps)
        elif self.schedule == "cosine":
            # Cosine schedule (Improved DDPM)
            s = 0.008
            steps = torch.arange(self.num_steps + 1)
            alphas_cumprod = torch.cos(
                ((steps / self.num_steps) + s) / (1 + s) * torch.tensor(np.pi) * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif self.schedule == "sqrt":
            self.betas = torch.sqrt(torch.linspace(0.0001, 0.02, self.num_steps))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        # Compute cumulative product
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def get_timestep(self, t: int) -> Tuple[float, float, float]:
        """
        Get diffusion parameters at timestep t.
        Returns: (alpha_cumprod, sqrt_alpha, sqrt_one_minus_alpha)
        """
        alpha = self.alphas_cumprod[t].item()
        sqrt_alpha = self.sqrt_alphas_cumprod[t].item()
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].item()
        return alpha, sqrt_alpha, sqrt_one_minus_alpha


class LevyFlightSampler:
    """
    Isotropic α-stable Lévy flight noise sampler.
    
    For α-stable distribution with index α:
      P(z) ~ |z|^{-(1+α)} for large |z|
    
    Implementation using Chambers-Mallows-Stuck method.
    """
    
    def __init__(self, alpha: float, scale: float = 1.0, device: str = "cpu"):
        """
        Args:
            alpha: Lévy index ∈ (1, 2], higher = lighter tails
            scale: Scale parameter γ
            device: torch device
        """
        if not (1 < alpha <= 2):
            raise ValueError(f"Lévy index α must be in (1, 2], got {alpha}")
        
        self.alpha = alpha
        self.scale = scale
        self.device = device
        
        # Pre-compute constant
        self.c = (scipy_gamma(0.5) / scipy_gamma(self.alpha / 2.0)) ** (1.0 / self.alpha)
    
    def sample(self, shape: Tuple[int, ...], truncate: bool = False, 
               threshold: float = 10.0) -> torch.Tensor:
        """
        Sample isotropic α-stable noise.
        
        Args:
            shape: Output tensor shape
            truncate: Whether to apply tempered Lévy (truncation)
            threshold: Truncation threshold (units of σ)
        
        Returns:
            Noise tensor of shape `shape`
        """
        # Chambers-Mallows-Stuck method
        u = torch.randn(shape, device=self.device)
        v = torch.randn(shape, device=self.device)
        
        # Standard Gaussian for uniform angular distribution
        zeta = (np.pi / 2.0) * v
        
        # Stable subordinator
        w = -torch.log(torch.rand(shape, device=self.device))
        
        # Lévy sample (unnormalized)
        sin_zeta = torch.sin(zeta)
        cos_zeta = torch.cos(zeta)
        
        sin_alpha_zeta = torch.sin(self.alpha * zeta)
        cos_alpha_zeta = torch.cos(self.alpha * zeta)
        
        # Compute Lévy sample
        log_term = torch.log(w * cos_alpha_zeta / sin_zeta ** self.alpha)
        z = (sin_alpha_zeta / sin_zeta) * torch.exp((1.0 - self.alpha) / self.alpha * log_term)
        
        # Normalize and scale
        z = self.scale * self.c * z
        
        # Optional tempered Lévy (truncate extreme values)
        if truncate:
            z = torch.clamp(z, -threshold * self.scale, threshold * self.scale)
        
        return z


class EffectiveMediumEstimator:
    """
    Estimate effective operator and disorder covariance from ensemble.
    
    Phase 1 of Algorithm 2: offline computation of Aeff and Σmodel.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.aeff = None
        self.sigma_model = None
        self.lambda_eff = None
    
    def estimate_from_ensemble(self, operators: List[torch.Tensor],
                               prior_samples: torch.Tensor,
                               measurement_noise_std: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate effective operator and residual covariance.
        
        Args:
            operators: List of K operator matrices A(ω_k) ∈ R^{m × n}
            prior_samples: Probe signals z_j ~ p_prior(x), shape (J, n)
            measurement_noise_std: σ_y for measurement noise
        
        Returns:
            aeff: Effective operator (m, n)
            sigma_eff: Effective noise covariance (m, m)
            lambda_eff: Inverse of sigma_eff for likelihood computation
        """
        K = len(operators)
        device = self.device
        
        # Phase 1: Compute effective operator
        # Aeff = (1/K) Σ A(ω_k)
        operators_tensor = torch.stack(operators, dim=0)  # (K, m, n)
        self.aeff = torch.mean(operators_tensor, dim=0)  # (m, n)
        
        # Phase 1b: Estimate residual covariance from probe signals
        # Σmodel = Cov{(A(ω_k) - Aeff) z_j}
        J = prior_samples.shape[0]
        residuals = []
        
        for k in range(K):
            delta_a = operators[k] - self.aeff  # (m, n)
            residual_k = torch.matmul(delta_a, prior_samples.T)  # (m, J)
            residuals.append(residual_k)
        
        residuals = torch.cat(residuals, dim=1)  # (m, K*J)
        
        # Empirical covariance
        residuals_centered = residuals - residuals.mean(dim=1, keepdim=True)
        sigma_model = (residuals_centered @ residuals_centered.T) / (K * J - 1)
        
        # Effective noise covariance: Σeff = σ_y^2 I + Σmodel
        m = self.aeff.shape[0]
        sigma_y_sq = measurement_noise_std ** 2
        self.sigma_eff = sigma_y_sq * torch.eye(m, device=device) + sigma_model
        
        # Compute inverse for likelihood score
        self.lambda_eff = torch.linalg.inv(self.sigma_eff)  # (m, m)
        
        return self.aeff, self.sigma_eff, self.lambda_eff
    
    def estimate_from_analytic(self, aeff: torch.Tensor,
                               sigma_model: torch.Tensor,
                               measurement_noise_std: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use analytically provided effective operator and model covariance.
        
        Args:
            aeff: Effective operator
            sigma_model: Model covariance from operator disorder
            measurement_noise_std: σ_y for measurement noise
        
        Returns:
            aeff, sigma_eff, lambda_eff (same as estimate_from_ensemble)
        """
        device = self.device
        self.aeff = aeff.to(device)
        sigma_model = sigma_model.to(device)
        
        m = self.aeff.shape[0]
        sigma_y_sq = measurement_noise_std ** 2
        self.sigma_eff = sigma_y_sq * torch.eye(m, device=device) + sigma_model
        self.lambda_eff = torch.linalg.inv(self.sigma_eff)
        
        return self.aeff, self.sigma_eff, self.lambda_eff


class DisorderAveragedALSS:
    """
    Disorder-Adaptive Anomalous Langevin Score Sampling.
    
    Algorithm 2 from TPAMI draft:
      Phase 1: Offline effective medium estimation
      Phase 2: Online anomalous diffusion sampling with disorder-averaged likelihood
    """
    
    def __init__(self, score_network: nn.Module, config: DisorderConfig,
                 device: str = "cpu"):
        """
        Args:
            score_network: Neural network s_θ(x, t) predicting ∇_x log p_t(x)
            config: DisorderConfig with algorithm parameters
            device: torch device
        """
        self.device = device
        self.score_network = score_network.to(device)
        self.config = config
        
        # Initialize components
        self.beta_scheduler = BetaScheduler(config.num_steps, config.beta_schedule)
        self.levy_sampler = LevyFlightSampler(
            alpha=config.levy_alpha,
            scale=config.levy_gamma,
            device=device
        )
        self.em_estimator = EffectiveMediumEstimator(device=device)
        
        # Cache for effective medium (Phase 1)
        self.aeff = None
        self.lambda_eff = None
    
    def setup_disorder_averaging(self, operators: Optional[List[torch.Tensor]] = None,
                                 aeff: Optional[torch.Tensor] = None,
                                 sigma_model: Optional[torch.Tensor] = None,
                                 prior_samples: Optional[torch.Tensor] = None,
                                 measurement_noise_std: float = 1.0) -> None:
        """
        Phase 1: Offline effective medium estimation.
        
        Provide EITHER:
          - `operators` + `prior_samples` for empirical estimation
          - `aeff` + `sigma_model` for analytic specification
        
        Args:
            operators: List of K random operators {A(ω_k)}
            aeff: Analytically provided effective operator
            sigma_model: Analytically provided model covariance
            prior_samples: Probe signals for empirical estimation
            measurement_noise_std: σ_y parameter
        """
        if operators is not None and prior_samples is not None:
            # Empirical estimation from ensemble
            self.em_estimator.estimate_from_ensemble(
                operators, prior_samples, measurement_noise_std
            )
        elif aeff is not None and sigma_model is not None:
            # Analytic specification
            self.em_estimator.estimate_from_analytic(
                aeff, sigma_model, measurement_noise_std
            )
        else:
            raise ValueError(
                "Must provide either (operators, prior_samples) "
                "or (aeff, sigma_model)"
            )
        
        self.aeff = self.em_estimator.aeff
        self.lambda_eff = self.em_estimator.lambda_eff
    
    def compute_likelihood_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihood score ∇_x log p(y|x).
        
        For disorder-averaged likelihood:
          p(y|x) ≈ N(y; Aeff x, Σeff)
          ∇_x log p(y|x) = Aeff^T Λeff (y - Aeff x)
        
        Args:
            x: Current sample (batch_size, n)
            y: Measurement (batch_size, m)
        
        Returns:
            Likelihood score (batch_size, n)
        """
        if self.aeff is None or self.lambda_eff is None:
            raise RuntimeError("Call setup_disorder_averaging() first")
        
        # Residual
        residual = y - torch.matmul(x, self.aeff.T)  # (batch, m)
        
        # Likelihood score: Aeff^T Λeff residual
        score = torch.matmul(residual, self.lambda_eff.T @ self.aeff)  # (batch, n)
        
        return score
    
    def sample(self, y: torch.Tensor,
               num_samples: int = 1,
               verbose: bool = False) -> torch.Tensor:
        """
        Phase 2: Online anomalous sampling via reverse diffusion.
        
        Algorithm 2, lines 9-21:
          For i = N down to 1:
            1. Disorder-averaged gradient (prior + likelihood)
            2. Lévy flight noise injection
            3. Update: x_{i-1} = x_i + η_i d_i + γ_i z
        
        Args:
            y: Measurements (batch_size, m) or (m,)
            num_samples: Number of posterior samples to generate
            verbose: Print progress
        
        Returns:
            Reconstructed signals (num_samples, batch_size, n)
        """
        if self.aeff is None:
            raise RuntimeError("Call setup_disorder_averaging() first")
        
        # Handle single measurement
        if y.dim() == 1:
            y = y.unsqueeze(0)
        batch_size, m = y.shape
        n = self.aeff.shape[1]
        
        y = y.to(self.device)
        
        # Initialize from high-temperature noise
        x = torch.randn(num_samples, batch_size, n, device=self.device)
        
        # Reverse diffusion loop
        for i in range(self.config.num_steps - 1, -1, -1):
            # Step size and noise scale (can be customized)
            eta = 0.1 / np.sqrt(self.config.num_steps)  # Step size
            gamma = np.sqrt(eta)  # Noise scale
            
            # Get time-dependent noise level
            alpha_cumprod, sqrt_alpha, sqrt_one_minus_alpha = self.beta_scheduler.get_timestep(i)
            
            # Repeat measurement for batch dimension
            y_batch = y.unsqueeze(0).repeat(num_samples, 1, 1)  # (num_samples, batch, m)
            
            # 1. Disorder-averaged gradient
            # Prior score from learned network
            t_tensor = torch.tensor([i], device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                # Reshape for network: (num_samples * batch, n)
                x_flat = x.reshape(num_samples * batch_size, n)
                t_expanded = t_tensor.expand(num_samples * batch_size)
                
                prior_score_flat = self.score_network(x_flat, t_expanded)
                prior_score = prior_score_flat.reshape(num_samples, batch_size, n)
            
            # Likelihood score
            likelihood_score = self.compute_likelihood_score(
                x.reshape(num_samples * batch_size, n),
                y_batch.reshape(num_samples * batch_size, m)
            ).reshape(num_samples, batch_size, n)
            
            # Combined drift
            d = prior_score + self.config.likelihood_weight * likelihood_score
            
            # 2. Lévy flight injection
            z = self.levy_sampler.sample(
                (num_samples, batch_size, n),
                truncate=self.config.use_tempered_levy,
                threshold=self.config.levy_truncation_threshold
            )
            
            # 3. Update
            x = x + eta * d + gamma * z
            
            if verbose and (i % max(1, self.config.num_steps // 10) == 0):
                print(f"  Step {i}: mean(|x|) = {x.mean().abs().item():.4f}")
        
        return x
    
    def sample_with_consistency_regularization(self, y: torch.Tensor,
                                                num_samples: int = 1,
                                                verbose: bool = False) -> torch.Tensor:
        """
        Sampling with optional consistency regularization.
        
        Adds penalty term: λ_reg ∥A x - y∥_2^2 to likelihood score.
        
        Args:
            y: Measurements
            num_samples: Number of posterior samples
            verbose: Print progress
        
        Returns:
            Reconstructed signals with regularization
        """
        # Similar to sample() but with additional consistency term
        # For brevity, implemented as wrapper with modified likelihood weight
        
        if self.config.consistency_reg_weight <= 0:
            return self.sample(y, num_samples, verbose)
        
        # Store original weight
        original_weight = self.config.likelihood_weight
        
        # Enhanced likelihood weight includes consistency
        self.config.likelihood_weight = original_weight * (1 + self.config.consistency_reg_weight)
        
        try:
            result = self.sample(y, num_samples, verbose)
        finally:
            self.config.likelihood_weight = original_weight
        
        return result


class SinglePixelImagingALSS:
    """
    Specialized interface for single-pixel imaging with sequential measurement.
    
    Integrates disorder-averaging with sequential measurement design for SPI.
    Supports e-value based hypothesis testing (ICLR paper).
    """
    
    def __init__(self, score_network: nn.Module, config: DisorderConfig,
                 device: str = "cpu"):
        """
        Args:
            score_network: Neural network for SPI
            config: DisorderConfig
            device: torch device
        """
        self.alss = DisorderAveragedALSS(score_network, config, device)
        self.device = device
        self.measurement_history = []
        self.pattern_history = []
    
    def add_measurement(self, measurement: float, pattern: torch.Tensor) -> None:
        """
        Add single-pixel measurement.
        
        Args:
            measurement: Scalar measurement y = <h, x> + noise
            pattern: Binary pattern h ∈ {0,1}^M
        """
        self.measurement_history.append(measurement)
        self.pattern_history.append(pattern.clone().to(self.device))
    
    def get_measurement_matrix(self) -> torch.Tensor:
        """Get accumulated measurement matrix H_N from history."""
        if not self.pattern_history:
            raise RuntimeError("No measurements collected")
        
        H = torch.stack(self.pattern_history, dim=0)  # (N, M)
        return H
    
    def compute_likelihood_ratio_evalue(self, x_recon: torch.Tensor,
                                        h0_params: Dict,
                                        h1_params: Dict) -> torch.Tensor:
        """
        Compute likelihood ratio e-value for hypothesis testing.
        
        E_LR = p(y|H1) / p(y|H0)
        
        Args:
            x_recon: Reconstructed image (M,)
            h0_params: Parameters for H0 distribution (e.g., {"mean": μ0, "cov": Σ0})
            h1_params: Parameters for H1 distribution
        
        Returns:
            E-value (scalar)
        """
        y_vec = torch.tensor(self.measurement_history, device=self.device)
        H = self.get_measurement_matrix()
        
        # Predicted measurements
        y_pred = torch.matmul(H, x_recon.unsqueeze(-1)).squeeze(-1)
        
        # Log likelihood ratio
        mu_0 = h0_params.get("mean", torch.zeros_like(y_pred))
        mu_1 = h1_params.get("mean", torch.zeros_like(y_pred))
        
        sigma_0 = h0_params.get("cov", torch.eye(len(y_vec), device=self.device))
        sigma_1 = h1_params.get("cov", torch.eye(len(y_vec), device=self.device))
        
        log_det_ratio = torch.logdet(sigma_0) - torch.logdet(sigma_1)
        
        residual_0 = y_vec - mu_0
        residual_1 = y_vec - mu_1
        
        mahal_0 = torch.sum(residual_0 * torch.linalg.solve(sigma_0, residual_0))
        mahal_1 = torch.sum(residual_1 * torch.linalg.solve(sigma_1, residual_1))
        
        log_evalue = 0.5 * (log_det_ratio + mahal_1 - mahal_0)
        evalue = torch.exp(log_evalue)
        
        return evalue
    
    def sequential_test(self, threshold: float = 20.0, h0_params: Optional[Dict] = None,
                       h1_params: Optional[Dict] = None) -> bool:
        """
        Anytime-valid hypothesis test using e-values.
        
        Args:
            threshold: E-value threshold for rejection (e.g., 20 for α ≈ 0.05)
            h0_params: H0 distribution parameters
            h1_params: H1 distribution parameters
        
        Returns:
            True if H0 should be rejected, False otherwise
        """
        if not self.measurement_history:
            return False
        
        # Estimate current posterior
        y_vec = torch.tensor(self.measurement_history, dtype=torch.float32, device=self.device)
        
        # Simple point estimate (posterior mean approximation)
        H = self.get_measurement_matrix()
        try:
            x_ls = torch.linalg.lstsq(H, y_vec).solution
        except:
            x_ls = torch.zeros(H.shape[1], device=self.device)
        
        # Compute e-value
        if h0_params is None:
            h0_params = {"mean": torch.zeros_like(y_vec), "cov": torch.eye(len(y_vec), device=self.device)}
        if h1_params is None:
            h1_params = {"mean": torch.ones_like(y_vec), "cov": torch.eye(len(y_vec), device=self.device)}
        
        evalue = self.compute_likelihood_ratio_evalue(x_ls, h0_params, h1_params)
        
        return evalue.item() > threshold


def main_example():
    """Example usage of DisorderAveragedALSS."""
    
    print("=" * 70)
    print("Disorder-Averaged Anomalous Langevin Score Sampling (DA-ALSS)")
    print("=" * 70)
    
    # Configuration
    config = DisorderConfig(
        num_steps=100,
        beta_schedule="cosine",
        levy_alpha=1.5,
        num_disorder_samples=50,
        measurement_noise_std=0.1
    )
    
    # Problem setup
    m, n = 64, 256  # m measurements, n signal dimension
    batch_size = 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create dummy score network
    class SimpleScoreNet(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )
        
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """x: (batch, n), t: (batch,)"""
            t_emb = t.unsqueeze(-1).float() / 1000.0
            x_and_t = torch.cat([x, t_emb], dim=-1)
            return self.net(x_and_t)
    
    score_net = SimpleScoreNet(n).to(device)
    
    # Create sampler
    sampler = DisorderAveragedALSS(score_net, config, device=device)
    
    # Setup disorder averaging (Phase 1)
    print("Phase 1: Effective Medium Estimation")
    print("-" * 70)
    
    # Create ensemble of random operators
    operators = [
        torch.randn(m, n, device=device) / np.sqrt(m)
        for _ in range(config.num_disorder_samples)
    ]
    
    # Probe signals
    prior_samples = torch.randn(16, n, device=device)
    
    sampler.setup_disorder_averaging(
        operators=operators,
        prior_samples=prior_samples,
        measurement_noise_std=config.measurement_noise_std
    )
    print(f"  Aeff shape: {sampler.aeff.shape}")
    print(f"  Λeff shape: {sampler.lambda_eff.shape}")
    
    # Generate measurement
    print("\nPhase 2: Anomalous Diffusion Sampling")
    print("-" * 70)
    
    x_true = torch.randn(batch_size, n, device=device)
    y = torch.matmul(x_true, sampler.aeff.T) + config.measurement_noise_std * torch.randn(batch_size, m, device=device)
    
    print(f"  True signal shape: {x_true.shape}")
    print(f"  Measurement shape: {y.shape}")
    
    # Sample from posterior
    num_samples = 5
    x_samples = sampler.sample(y, num_samples=num_samples, verbose=True)
    
    print(f"\nPosterior samples shape: {x_samples.shape}")
    print(f"  Expected: ({num_samples}, {batch_size}, {n})")
    
    # Compute PSNR
    x_mean = x_samples.mean(dim=0)
    mse = F.mse_loss(x_mean, x_true)
    psnr = 10 * np.log10(1.0 / (mse.item() + 1e-8))
    print(f"\nPSNR (posterior mean vs. true): {psnr:.2f} dB")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
