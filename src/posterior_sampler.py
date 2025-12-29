"""
Advanced Posterior Sampling and Inference Methods

Implements:
  1. Markov Chain Monte Carlo (MCMC) sampling
  2. Variational inference and reparameterization tricks
  3. Score-based diffusion sampling for posteriors
  4. Hamiltonian Monte Carlo for efficient exploration
  5. Sequential and adaptive posterior updates

References:
  - TPAMI paper: Section 2.2-2.3 on posterior inference
  - ICLR paper: Section 3.2 on score-based diffusion sampling
  - "Diffusion Models as a Unified Framework" (Song et al., 2021)
  - Bayesian inference and uncertainty quantification literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class SamplingMethod(Enum):
    """Types of posterior sampling methods."""
    HMC = "hmc"                        # Hamiltonian Monte Carlo
    NUTS = "nuts"                      # No-U-Turn Sampler
    RWMH = "rwmh"                      # Random walk Metropolis-Hastings
    MALA = "mala"                      # Manifold adjusted Langevin algorithm
    ADVI = "advi"                      # Automatic differentiation VI
    SCORE_DIFFUSION = "score_diffusion"  # Score-based diffusion
    LAPLACE = "laplace"                # Laplace approximation
    PARTICLE_FILTER = "particle_filter"  # Sequential particle filtering


@dataclass
class PosteriorConfig:
    """Configuration for posterior sampling."""
    # Sampling method
    method: SamplingMethod = SamplingMethod.HMC
    
    # MCMC parameters
    num_samples: int = 1000            # Total posterior samples
    num_warmup: int = 500              # Burn-in steps
    num_chains: int = 4                # Parallel chains
    
    # Proposal/step size
    step_size: float = 0.01            # Integration/proposal step size
    num_steps: int = 20                # Steps per iteration (HMC)
    
    # Adaptation
    adapt_step_size: bool = True       # Adaptive step size (dual averaging)
    target_acceptance: float = 0.8     # Target acceptance rate
    
    # Variational inference
    num_variational_samples: int = 100 # VI approximation samples
    num_vi_iterations: int = 1000      # VI optimization steps
    vi_learning_rate: float = 0.01     # VI optimization rate
    
    # Score-based diffusion
    num_diffusion_steps: int = 100     # Number of diffusion steps
    diffusion_schedule: str = "linear"  # "linear", "cosine", "sqrt"
    
    # Convergence
    target_ess: float = 0.5            # Target effective sample size ratio
    max_iterations: int = 2000         # Max iterations before stopping
    
    # Diagnostics
    compute_diagnostics: bool = True   # Compute R-hat, ESS, etc.
    verbose: bool = False


@dataclass
class PosteriorSamples:
    """Container for posterior samples."""
    samples: torch.Tensor              # (num_samples, dim)
    log_weights: Optional[torch.Tensor] = None  # For importance sampling
    log_prob: Optional[torch.Tensor] = None     # Log posterior for each sample
    
    # Statistics
    mean: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    
    # Diagnostics
    r_hat: Optional[float] = None      # Gelman-Rubin convergence diagnostic
    ess_ratio: Optional[float] = None  # Effective sample size ratio
    acceptance_rate: Optional[float] = None
    
    def compute_statistics(self) -> None:
        """Compute posterior statistics."""
        if self.log_weights is not None:
            # Importance-weighted statistics
            weights = torch.softmax(self.log_weights, dim=0)
            self.mean = torch.sum(self.samples * weights.unsqueeze(-1), dim=0)
            centered = self.samples - self.mean.unsqueeze(0)
            self.covariance = torch.einsum('ni,nj->ij', centered * weights.unsqueeze(-1), centered)
            self.std = torch.sqrt(torch.diag(self.covariance))
        else:
            # Standard statistics
            self.mean = torch.mean(self.samples, dim=0)
            self.std = torch.std(self.samples, dim=0)
            centered = self.samples - self.mean.unsqueeze(0)
            self.covariance = (centered.T @ centered) / (self.samples.shape[0] - 1)


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampling."""
    
    @staticmethod
    def random_walk_mh(log_posterior: Callable, initial_state: torch.Tensor,
                      step_size: float, num_samples: int = 1000,
                      num_warmup: int = 500) -> Tuple[torch.Tensor, float]:
        """
        Random walk Metropolis-Hastings.
        
        Proposal: q(x*|x) = N(x*, x + σ²I)
        
        Args:
            log_posterior: Log posterior density function
            initial_state: Starting point
            step_size: Proposal standard deviation
            num_samples: Number of samples
            num_warmup: Burn-in iterations
        
        Returns:
            (samples, acceptance_rate): Posterior samples and MH acceptance rate
        """
        device = initial_state.device
        dim = initial_state.shape[0]
        
        samples = torch.zeros(num_warmup + num_samples, dim, device=device)
        samples[0] = initial_state
        
        current_state = initial_state.clone()
        current_log_prob = log_posterior(current_state)
        
        num_accepted = 0
        
        for i in range(1, num_warmup + num_samples):
            # Proposal: random walk
            proposal = current_state + step_size * torch.randn(dim, device=device)
            proposal_log_prob = log_posterior(proposal)
            
            # Metropolis-Hastings ratio (symmetric proposal)
            log_alpha = proposal_log_prob - current_log_prob
            
            # Accept/reject
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                current_state = proposal
                current_log_prob = proposal_log_prob
                num_accepted += 1
            
            samples[i] = current_state
        
        # Return post-warmup samples
        acceptance_rate = num_accepted / (num_warmup + num_samples)
        return samples[num_warmup:], acceptance_rate
    
    @staticmethod
    def mala(log_posterior: Callable, grad_log_posterior: Callable,
            initial_state: torch.Tensor, step_size: float,
            num_samples: int = 1000, num_warmup: int = 500) -> Tuple[torch.Tensor, float]:
        """
        Manifold Adjusted Langevin Algorithm (MALA).
        
        Proposal: x* = x + (σ²/2)∇log p(x) + σ ε, ε ~ N(0,I)
        
        Uses gradient information for better proposals.
        
        Args:
            log_posterior: Log posterior density
            grad_log_posterior: Gradient of log posterior
            initial_state: Starting point
            step_size: Integration step size
            num_samples: Number of samples
            num_warmup: Burn-in iterations
        
        Returns:
            (samples, acceptance_rate): Posterior samples and acceptance rate
        """
        device = initial_state.device
        dim = initial_state.shape[0]
        
        samples = torch.zeros(num_warmup + num_samples, dim, device=device)
        samples[0] = initial_state
        
        current_state = initial_state.clone().requires_grad_(True)
        current_log_prob = log_posterior(current_state)
        
        num_accepted = 0
        
        for i in range(1, num_warmup + num_samples):
            # Compute gradient
            current_log_prob_val = log_posterior(current_state.detach())
            grad = grad_log_posterior(current_state)
            
            # MALA proposal
            drift = (step_size / 2) * grad
            noise = np.sqrt(step_size) * torch.randn(dim, device=device)
            proposal = current_state.detach() + drift.detach() + noise
            
            # Acceptance probability
            proposal_log_prob = log_posterior(proposal)
            
            # Symmetric acceptance ratio (MALA specific)
            forward_term = drift @ (proposal - current_state.detach())
            backward_term = (grad_log_posterior(proposal) @ (current_state.detach() - proposal)) / 2
            
            log_alpha = proposal_log_prob - current_log_prob_val + backward_term + forward_term
            
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                current_state = proposal.requires_grad_(True)
                current_log_prob = proposal_log_prob
                num_accepted += 1
            else:
                current_state = current_state.detach().requires_grad_(True)
            
            samples[i] = current_state.detach()
        
        acceptance_rate = num_accepted / (num_warmup + num_samples)
        return samples[num_warmup:], acceptance_rate


class HamiltonianMonteCarlo:
    """Hamiltonian Monte Carlo sampling."""
    
    @staticmethod
    def hmc_step(log_posterior: Callable, grad_log_posterior: Callable,
                position: torch.Tensor, momentum: torch.Tensor,
                step_size: float, num_steps: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Single HMC step with leapfrog integrator.
        
        Args:
            log_posterior: Log posterior density
            grad_log_posterior: Gradient of log posterior
            position: Current position
            momentum: Current momentum
            step_size: Integration step size
            num_steps: Number of leapfrog steps
        
        Returns:
            (new_position, new_momentum, acceptance_prob): Updated state and probability
        """
        device = position.device
        
        # Initial kinetic energy
        initial_ke = 0.5 * torch.sum(momentum**2)
        initial_pe = -log_posterior(position)
        initial_energy = initial_ke + initial_pe
        
        # Leapfrog integration
        pos = position.clone().requires_grad_(True)
        mom = momentum.clone()
        
        # Half step for momentum
        grad = grad_log_posterior(pos)
        mom = mom + (step_size / 2) * grad
        
        # Full steps
        for _ in range(num_steps):
            pos = pos.detach() + step_size * mom
            pos.requires_grad_(True)
            
            grad = grad_log_posterior(pos)
            mom = mom + step_size * grad
        
        # Final half step
        grad = grad_log_posterior(pos)
        mom = mom + (step_size / 2) * grad
        
        # Final kinetic and potential energy
        final_ke = 0.5 * torch.sum(mom**2)
        final_pe = -log_posterior(pos.detach())
        final_energy = final_ke + final_pe
        
        # Metropolis acceptance
        log_alpha = -(final_energy - initial_energy)
        acceptance_prob = torch.min(torch.tensor(1.0, device=device), torch.exp(log_alpha))
        
        if torch.rand(1, device=device) < acceptance_prob:
            return pos.detach(), -mom, float(acceptance_prob)
        else:
            return position, momentum, float(acceptance_prob)
    
    @staticmethod
    def sample(log_posterior: Callable, grad_log_posterior: Callable,
              initial_state: torch.Tensor, config: PosteriorConfig) -> PosteriorSamples:
        """
        HMC sampling.
        
        Args:
            log_posterior: Log posterior density
            grad_log_posterior: Gradient of log posterior
            initial_state: Starting point
            config: PosteriorConfig
        
        Returns:
            PosteriorSamples
        """
        device = initial_state.device
        dim = initial_state.shape[0]
        
        samples = torch.zeros(config.num_samples, dim, device=device)
        log_probs = torch.zeros(config.num_samples, device=device)
        
        position = initial_state.clone()
        num_accepted = 0
        
        # Warmup phase
        for i in range(config.num_warmup):
            momentum = torch.randn(dim, device=device)
            position, momentum, accept_prob = HamiltonianMonteCarlo.hmc_step(
                log_posterior, grad_log_posterior, position, momentum,
                config.step_size, config.num_steps
            )
            num_accepted += (1 if accept_prob > 0.5 else 0)
        
        # Sampling phase
        for i in range(config.num_samples):
            momentum = torch.randn(dim, device=device)
            position, momentum, accept_prob = HamiltonianMonteCarlo.hmc_step(
                log_posterior, grad_log_posterior, position, momentum,
                config.step_size, config.num_steps
            )
            samples[i] = position
            log_probs[i] = log_posterior(position)
            num_accepted += (1 if accept_prob > 0.5 else 0)
        
        acceptance_rate = num_accepted / (config.num_warmup + config.num_samples)
        
        result = PosteriorSamples(samples=samples, log_prob=log_probs)
        result.acceptance_rate = acceptance_rate
        result.compute_statistics()
        
        return result


class VariationalInference:
    """Variational inference posterior approximation."""
    
    @staticmethod
    def mean_field_gaussian(log_posterior: Callable, initial_mean: torch.Tensor,
                           initial_log_std: Optional[torch.Tensor] = None,
                           config: Optional[PosteriorConfig] = None,
                           num_iterations: int = 1000) -> PosteriorSamples:
        """
        Mean-field Gaussian variational inference.
        
        q(x) = N(μ, diag(σ²))
        
        Optimize: ELBO = E_q[log p(x|y)] - KL(q||p)
        
        Args:
            log_posterior: Log posterior density p(x|y)
            initial_mean: Initial mean estimate
            initial_log_std: Initial log standard deviation
            config: PosteriorConfig
            num_iterations: VI iterations
        
        Returns:
            PosteriorSamples
        """
        device = initial_mean.device
        dim = initial_mean.shape[0]
        
        if config is None:
            config = PosteriorConfig()
        
        # Initialize variational parameters
        mu = initial_mean.clone().requires_grad_(True)
        if initial_log_std is None:
            log_std = torch.zeros(dim, device=device, requires_grad=True)
        else:
            log_std = initial_log_std.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([mu, log_std], lr=config.vi_learning_rate)
        
        # VI optimization
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Reparameterization trick
            eps = torch.randn(config.num_variational_samples, dim, device=device)
            std = torch.exp(log_std)
            samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
            
            # ELBO: E[log p] - KL
            log_prob = torch.stack([log_posterior(s) for s in samples])
            
            # KL divergence: KL(q||p) ≈ -0.5 * ∑ (1 + log σ² - μ² - σ²)
            kl_div = -0.5 * torch.sum(1 + 2*log_std - mu**2 - torch.exp(2*log_std))
            
            # ELBO
            elbo = torch.mean(log_prob) + kl_div
            loss = -elbo  # Minimize negative ELBO
            
            loss.backward()
            optimizer.step()
            
            if config.verbose and iteration % 100 == 0:
                print(f"VI iter {iteration}: ELBO = {elbo.item():.4f}")
        
        # Generate samples from learned q
        with torch.no_grad():
            std = torch.exp(log_std)
            eps = torch.randn(config.num_samples, dim, device=device)
            samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        
        result = PosteriorSamples(samples=samples)
        result.compute_statistics()
        
        return result


class ScoreDiffusionSampler:
    """Score-based diffusion sampling from posteriors."""
    
    @staticmethod
    def score_function(x: torch.Tensor, t: torch.Tensor, score_net: nn.Module,
                      likelihood_fn: Callable, y: torch.Tensor,
                      noise_scale: float = 1.0) -> torch.Tensor:
        """
        Compute score function for posterior diffusion.
        
        ∇_x log p(x|y) = ∇_x log p(y|x) + ∇_x log p(x)
        
        Args:
            x: Current state
            t: Time variable (0 to 1)
            score_net: Learned score network ∇_x log p_t(x)
            likelihood_fn: Function computing ∇_x log p(y|x)
            y: Observed data
            noise_scale: Noise scale factor
        
        Returns:
            Score vector
        """
        # Prior score (learned)
        prior_score = score_net(x.unsqueeze(0), t).squeeze(0)
        
        # Likelihood score
        likelihood_score = likelihood_fn(x, y)
        
        # Total score
        score = prior_score + noise_scale * likelihood_score
        
        return score
    
    @staticmethod
    def sample(score_net: nn.Module, likelihood_fn: Callable, y: torch.Tensor,
              initial_x: torch.Tensor, config: PosteriorConfig,
              temperature: float = 1.0) -> PosteriorSamples:
        """
        Sample from posterior using score-based diffusion.
        
        Reverse SDE: dx = [f(t) x + g(t)² ∇_x log p(x,y)] dt + g(t) dw
        
        Args:
            score_net: Trained score network
            likelihood_fn: Function computing likelihood score
            y: Observed data
            initial_x: Initial sample (pure noise)
            config: PosteriorConfig
            temperature: Temperature for score scaling
        
        Returns:
            PosteriorSamples
        """
        device = initial_x.device
        
        samples = []
        
        for sample_idx in range(config.num_samples):
            x = initial_x.clone()
            
            # Reverse diffusion (from t=1 to t=0)
            dt = 1.0 / config.num_diffusion_steps
            
            for step in range(config.num_diffusion_steps):
                t = 1.0 - (step + 1) * dt
                t_tensor = torch.tensor(t, device=device)
                
                # Score
                score = ScoreDiffusionSampler.score_function(
                    x, t_tensor, score_net, likelihood_fn, y
                )
                
                # Langevin step
                x = x + (dt / 2) * score + np.sqrt(dt) * temperature * torch.randn_like(x)
            
            samples.append(x)
        
        samples = torch.stack(samples)
        
        result = PosteriorSamples(samples=samples)
        result.compute_statistics()
        
        return result


class LaplaceApproximation:
    """Laplace approximation for fast posterior inference."""
    
    @staticmethod
    def approximate(log_posterior: Callable, grad_log_posterior: Callable,
                   mode: torch.Tensor, config: PosteriorConfig) -> PosteriorSamples:
        """
        Laplace approximation: p(x|y) ≈ N(x* | -H⁻¹)
        
        where x* is the MAP estimate and H is the Hessian.
        
        Args:
            log_posterior: Log posterior
            grad_log_posterior: Gradient of log posterior
            mode: MAP estimate (mode of posterior)
            config: PosteriorConfig
        
        Returns:
            PosteriorSamples (Gaussian approximation)
        """
        device = mode.device
        dim = mode.shape[0]
        
        # Compute Hessian numerically
        hessian = torch.zeros(dim, dim, device=device)
        eps = 1e-4
        
        grad_at_mode = grad_log_posterior(mode)
        
        for i in range(dim):
            mode_plus = mode.clone()
            mode_plus[i] += eps
            grad_plus = grad_log_posterior(mode_plus)
            
            mode_minus = mode.clone()
            mode_minus[i] -= eps
            grad_minus = grad_log_posterior(mode_minus)
            
            hessian[i] = (grad_plus - grad_minus) / (2 * eps)
        
        # Negative Hessian is precision matrix
        precision = -hessian
        
        try:
            cov = torch.linalg.inv(precision)
        except:
            # Regularize if singular
            precision = precision + 1e-6 * torch.eye(dim, device=device)
            cov = torch.linalg.inv(precision)
        
        # Sample from Gaussian approximation
        L = torch.linalg.cholesky(cov)
        z = torch.randn(config.num_samples, dim, device=device)
        samples = mode.unsqueeze(0) + z @ L.T
        
        result = PosteriorSamples(samples=samples)
        result.compute_statistics()
        
        return result


class PosteriorSampler:
    """Main posterior sampling interface."""
    
    def __init__(self, config: PosteriorConfig):
        """Initialize sampler."""
        self.config = config
    
    def sample(self, log_posterior: Callable,
              grad_log_posterior: Optional[Callable] = None,
              initial_state: Optional[torch.Tensor] = None,
              **kwargs) -> PosteriorSamples:
        """
        Sample from posterior.
        
        Args:
            log_posterior: Log posterior density
            grad_log_posterior: Gradient (required for HMC, MALA)
            initial_state: Starting point
            **kwargs: Additional arguments
        
        Returns:
            PosteriorSamples
        """
        if self.config.method == SamplingMethod.HMC:
            assert grad_log_posterior is not None, "HMC requires gradient"
            assert initial_state is not None, "HMC requires initial state"
            return HamiltonianMonteCarlo.sample(
                log_posterior, grad_log_posterior, initial_state, self.config
            )
        
        elif self.config.method == SamplingMethod.RWMH:
            assert initial_state is not None
            samples, accept_rate = MetropolisHastings.random_walk_mh(
                log_posterior, initial_state,
                self.config.step_size, self.config.num_samples, self.config.num_warmup
            )
            result = PosteriorSamples(samples=samples)
            result.acceptance_rate = accept_rate
            result.compute_statistics()
            return result
        
        elif self.config.method == SamplingMethod.MALA:
            assert grad_log_posterior is not None
            assert initial_state is not None
            samples, accept_rate = MetropolisHastings.mala(
                log_posterior, grad_log_posterior, initial_state,
                self.config.step_size, self.config.num_samples, self.config.num_warmup
            )
            result = PosteriorSamples(samples=samples)
            result.acceptance_rate = accept_rate
            result.compute_statistics()
            return result
        
        elif self.config.method == SamplingMethod.ADVI:
            return VariationalInference.mean_field_gaussian(
                log_posterior, initial_state or torch.zeros(kwargs.get('dim', 10)),
                config=self.config
            )
        
        elif self.config.method == SamplingMethod.LAPLACE:
            assert grad_log_posterior is not None
            assert initial_state is not None
            return LaplaceApproximation.approximate(
                log_posterior, grad_log_posterior, initial_state, self.config
            )
        
        else:
            raise ValueError(f"Unknown sampling method: {self.config.method}")


def main_example():
    """Example demonstrating posterior sampling."""
    
    print("=" * 70)
    print("Advanced Posterior Sampling and Inference Methods")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup synthetic posterior
    print("\n1. Define Posterior (Gaussian mixture example)")
    print("-" * 70)
    
    def log_posterior(x):
        """Bimodal Gaussian mixture posterior."""
        mode1 = torch.tensor([2.0, 2.0], device=device)
        mode2 = torch.tensor([-2.0, -2.0], device=device)
        
        dist1 = torch.exp(-torch.sum((x - mode1)**2) / 2)
        dist2 = torch.exp(-torch.sum((x - mode2)**2) / 2)
        
        return torch.log(dist1 + dist2 + 1e-8)
    
    def grad_log_posterior(x):
        """Gradient of log posterior."""
        x_req = x.clone().requires_grad_(True)
        log_p = log_posterior(x_req)
        log_p.backward()
        return x_req.grad
    
    initial_state = torch.randn(2, device=device)
    
    # Test different samplers
    print("\n2. Hamiltonian Monte Carlo")
    print("-" * 70)
    
    config = PosteriorConfig(
        method=SamplingMethod.HMC,
        num_samples=500,
        num_warmup=200,
        step_size=0.1,
        num_steps=20,
        verbose=True
    )
    
    sampler = PosteriorSampler(config)
    hmc_result = sampler.sample(log_posterior, grad_log_posterior, initial_state)
    
    print(f"  HMC Acceptance Rate: {hmc_result.acceptance_rate:.2%}")
    print(f"  Posterior Mean: {hmc_result.mean}")
    print(f"  Posterior Std: {hmc_result.std}")
    
    print("\n3. Random Walk Metropolis-Hastings")
    print("-" * 70)
    
    config = PosteriorConfig(
        method=SamplingMethod.RWMH,
        num_samples=500,
        num_warmup=200,
        step_size=0.3,
        verbose=True
    )
    
    sampler = PosteriorSampler(config)
    rwmh_result = sampler.sample(log_posterior, None, initial_state)
    
    print(f"  RWMH Acceptance Rate: {rwmh_result.acceptance_rate:.2%}")
    print(f"  Posterior Mean: {rwmh_result.mean}")
    print(f"  Posterior Std: {rwmh_result.std}")
    
    print("\n4. MALA (Manifold Adjusted Langevin Algorithm)")
    print("-" * 70)
    
    config = PosteriorConfig(
        method=SamplingMethod.MALA,
        num_samples=500,
        num_warmup=200,
        step_size=0.1,
        verbose=True
    )
    
    sampler = PosteriorSampler(config)
    mala_result = sampler.sample(log_posterior, grad_log_posterior, initial_state)
    
    print(f"  MALA Acceptance Rate: {mala_result.acceptance_rate:.2%}")
    print(f"  Posterior Mean: {mala_result.mean}")
    print(f"  Posterior Std: {mala_result.std}")
    
    print("\n5. Variational Inference (Mean-Field Gaussian)")
    print("-" * 70)
    
    config = PosteriorConfig(
        method=SamplingMethod.ADVI,
        num_samples=500,
        num_variational_samples=100,
        num_vi_iterations=500,
        verbose=True
    )
    
    sampler = PosteriorSampler(config)
    vi_result = sampler.sample(log_posterior, None, initial_state)
    
    print(f"  VI Posterior Mean: {vi_result.mean}")
    print(f"  VI Posterior Std: {vi_result.std}")
    
    print("\n6. Laplace Approximation")
    print("-" * 70)
    
    # For Laplace, we need MAP estimate first
    from scipy.optimize import minimize
    
    def neg_log_posterior(x_np):
        x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
        return -log_posterior(x_torch).item()
    
    result = minimize(neg_log_posterior, initial_state.cpu().numpy(), method='BFGS')
    map_estimate = torch.tensor(result.x, device=device, dtype=torch.float32)
    
    config = PosteriorConfig(
        method=SamplingMethod.LAPLACE,
        num_samples=500,
        verbose=True
    )
    
    sampler = PosteriorSampler(config)
    laplace_result = sampler.sample(log_posterior, grad_log_posterior, map_estimate)
    
    print(f"  Laplace Posterior Mean: {laplace_result.mean}")
    print(f"  Laplace Posterior Std: {laplace_result.std}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
