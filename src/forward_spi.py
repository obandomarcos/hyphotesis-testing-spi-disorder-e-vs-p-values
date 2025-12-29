"""
Single-Pixel Imaging (SPI) Forward Models and Measurement Operators

Implements:
  1. Basic SPI measurement model: y = <h, x> + ε
  2. Random measurement patterns (Bernoulli, Hadamard, structured)
  3. Effective operator estimation for disorder-averaged framework
  4. Sequential measurement design with information gain
  5. Physical noise models (Poisson, Gaussian, mixed)

References:
  - Single-pixel imaging systems (computational sensing)
  - Disorder-averaging framework for random operators
  - ICLR paper: Section 2.1, 3.1 - SPI forward model and sequential measurement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
from scipy.linalg import hadamard
import warnings


class PatternType(Enum):
    """Types of measurement patterns for SPI."""
    RANDOM_BERNOULLI = "bernoulli"      # Random binary masks
    HADAMARD = "hadamard"              # Walsh-Hadamard basis
    GAUSSIAN = "gaussian"              # Random Gaussian patterns
    STRUCTURED = "structured"          # Structured binary patterns
    LEARNABLE = "learnable"            # Optimized via OED


class NoiseModel(Enum):
    """Types of physical noise in SPI."""
    GAUSSIAN = "gaussian"              # Additive Gaussian noise
    POISSON = "poisson"                # Photon shot noise
    MIXED = "mixed"                    # Gaussian + Poisson
    QUANTIZED = "quantized"            # Bit depth quantization


@dataclass
class SPIConfig:
    """Configuration for single-pixel imaging system."""
    # Image properties
    image_dim: int = 32                 # Image size: image_dim × image_dim
    num_pixels: int = 32 * 32           # Total pixels (auto-computed)
    
    # Measurement properties
    num_measurements: int = 256         # Number of SPI measurements
    pattern_type: PatternType = PatternType.RANDOM_BERNOULLI
    noise_model: NoiseModel = NoiseModel.GAUSSIAN
    
    # Noise parameters
    measurement_noise_std: float = 0.1  # σ for Gaussian noise
    photon_count: Optional[int] = None  # λ for Poisson noise
    quantization_bits: int = 16         # Bit depth for quantization
    
    # Disorder parameters (for disorder-averaging)
    enable_disorder: bool = True        # Model pattern disorder
    disorder_probability: float = 0.05  # Probability of random pattern flip
    num_disorder_samples: int = 50      # K: number of disorder realizations
    
    # Reconstruction task
    task: str = "reconstruction"        # "reconstruction", "classification", "detection"
    ground_truth_range: Tuple[float, float] = (0.0, 1.0)  # Image value range
    
    def __post_init__(self):
        """Auto-compute derived quantities."""
        self.num_pixels = self.image_dim ** 2


class SPIPatternGenerator:
    """Generate measurement patterns for single-pixel imaging."""
    
    def __init__(self, config: SPIConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.pattern_count = 0
        self.pattern_history = []
    
    def generate_random_bernoulli(self, num_patterns: Optional[int] = None) -> torch.Tensor:
        """
        Generate random binary patterns (Bernoulli distribution).
        
        Each pattern h ∈ {0, 1}^M with equal probability.
        
        Args:
            num_patterns: Number of patterns (if None, uses config.num_measurements)
        
        Returns:
            Patterns (num_patterns, image_dim, image_dim) or (num_patterns, num_pixels)
        """
        if num_patterns is None:
            num_patterns = self.config.num_measurements
        
        patterns = torch.randint(0, 2, (num_patterns, self.config.num_pixels),
                                 dtype=torch.float32, device=self.device)
        
        self.pattern_history.extend(patterns)
        self.pattern_count += num_patterns
        
        return patterns
    
    def generate_hadamard(self, num_patterns: Optional[int] = None) -> torch.Tensor:
        """
        Generate Walsh-Hadamard basis patterns.
        
        Deterministic orthogonal patterns based on Hadamard matrix.
        Provides structured, low-discrepancy sampling.
        
        Args:
            num_patterns: Number of patterns (if None, uses config.num_measurements)
        
        Returns:
            Hadamard patterns (num_patterns, num_pixels)
        """
        if num_patterns is None:
            num_patterns = self.config.num_measurements
        
        # Compute Hadamard matrix size (next power of 2)
        h_size = 2 ** int(np.ceil(np.log2(self.config.num_pixels)))
        H = torch.from_numpy(hadamard(h_size)).float()
        
        # Extract rows and reshape
        patterns = H[:num_patterns, :self.config.num_pixels]
        # Normalize to [0, 1]
        patterns = (patterns + 1) / 2
        
        patterns = patterns.to(self.device)
        self.pattern_history.extend(patterns)
        self.pattern_count += num_patterns
        
        return patterns
    
    def generate_gaussian(self, num_patterns: Optional[int] = None) -> torch.Tensor:
        """
        Generate Gaussian random patterns.
        
        Patterns h_n ~ N(0, I/M) for normalized variance.
        
        Args:
            num_patterns: Number of patterns
        
        Returns:
            Gaussian patterns (num_patterns, num_pixels)
        """
        if num_patterns is None:
            num_patterns = self.config.num_measurements
        
        # Standard normal normalized by sqrt(dim)
        patterns = torch.randn(num_patterns, self.config.num_pixels, device=self.device)
        patterns = patterns / np.sqrt(self.config.num_pixels)
        
        # Clip to valid range
        patterns = torch.clamp(patterns, 0, 1)
        
        self.pattern_history.extend(patterns)
        self.pattern_count += num_patterns
        
        return patterns
    
    def generate_structured(self, structure: str = "blocks") -> torch.Tensor:
        """
        Generate structured patterns (spatial or frequency-based).
        
        Args:
            structure: "blocks" (spatial blocks), "frequency" (frequency bands)
        
        Returns:
            Structured patterns (num_patterns, num_pixels)
        """
        num_patterns = self.config.num_measurements
        dim = self.config.image_dim
        
        patterns = []
        
        if structure == "blocks":
            # Divide image into blocks
            block_size = max(1, dim // int(np.sqrt(num_patterns)))
            for i in range(num_patterns):
                pattern = torch.zeros(dim, dim, device=self.device)
                block_i = (i * block_size) % dim
                block_j = ((i // dim) * block_size) % dim
                pattern[block_i:min(block_i+block_size, dim),
                        block_j:min(block_j+block_size, dim)] = 1.0
                patterns.append(pattern.flatten())
        
        elif structure == "frequency":
            # Frequency bands (DCT/frequency bases)
            for i in range(num_patterns):
                freq_level = i / num_patterns
                x = torch.arange(dim, device=self.device).float()
                y = torch.arange(dim, device=self.device).float()
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                # Sinusoidal pattern at frequency level
                pattern = torch.sin(2 * np.pi * freq_level * X / dim) * \
                         torch.cos(2 * np.pi * freq_level * Y / dim)
                pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
                patterns.append(pattern.flatten())
        
        patterns_tensor = torch.stack(patterns)
        self.pattern_history.extend(patterns_tensor)
        self.pattern_count += num_patterns
        
        return patterns_tensor
    
    def generate_adaptive(self, x_recon: torch.Tensor, score_fn: Callable) -> torch.Tensor:
        """
        Generate pattern optimized for next measurement (optimal experimental design).
        
        Uses information gain as criterion: h_next = argmax_h I(x; y_n | y_{1:n-1})
        
        Args:
            x_recon: Current reconstruction (posterior mean)
            score_fn: Function computing information gain for a pattern
        
        Returns:
            Optimal pattern (1, num_pixels)
        """
        # Generate candidates
        candidates = self.generate_random_bernoulli(num_patterns=10)
        
        # Evaluate information gain for each
        gains = []
        for h in candidates:
            gain = score_fn(h, x_recon)
            gains.append(gain.item() if torch.is_tensor(gain) else gain)
        
        # Select best
        best_idx = np.argmax(gains)
        best_pattern = candidates[best_idx:best_idx+1]
        
        self.pattern_history.append(best_pattern)
        self.pattern_count += 1
        
        return best_pattern


class SPIForwardModel:
    """
    Single-pixel imaging forward model.
    
    y = <h, x> + ε
    
    where:
      - x ∈ R^M: image (M = image_dim²)
      - h ∈ {0,1}^M: measurement pattern (binary or continuous)
      - ε: measurement noise
      - y ∈ R: scalar measurement
    """
    
    def __init__(self, config: SPIConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Pattern generator
        self.pattern_gen = SPIPatternGenerator(config, device)
        
        # Measurement history
        self.measurements = []
        self.patterns = []
        self.ground_truth = None
    
    def forward(self, x: torch.Tensor, h: torch.Tensor,
                noise_std: Optional[float] = None,
                return_components: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Compute single-pixel measurement.
        
        y = <h, x> + ε
        
        Args:
            x: Image (batch, num_pixels) or (num_pixels,)
            h: Pattern (batch, num_pixels) or (num_pixels,)
            noise_std: Noise standard deviation (if None, uses config)
            return_components: If True, return (measurement, clean_y, noise)
        
        Returns:
            Measurement y (scalar or batch) or tuple of components
        """
        # Handle batching
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        
        x = x.to(self.device).float()
        h = h.to(self.device).float()
        
        # Compute inner product
        if x.shape[0] != h.shape[0]:
            # Broadcast
            x = x.expand(h.shape[0], -1)
        
        y_clean = torch.sum(h * x, dim=-1)  # (batch,)
        
        # Add noise
        if noise_std is None:
            noise_std = self.config.measurement_noise_std
        
        noise = self._generate_noise(y_clean.shape, noise_std)
        y = y_clean + noise
        
        if return_components:
            return y, y_clean, noise
        return y
    
    def _generate_noise(self, shape: Tuple, noise_std: float) -> torch.Tensor:
        """Generate noise according to configured noise model."""
        if self.config.noise_model == NoiseModel.GAUSSIAN:
            return torch.randn(shape, device=self.device) * noise_std
        
        elif self.config.noise_model == NoiseModel.POISSON:
            # Poisson noise (photon counting)
            lambda_param = self.config.photon_count or max(1.0, 1.0 / (noise_std ** 2))
            return torch.poisson(torch.ones(shape, device=self.device) * lambda_param) - lambda_param
        
        elif self.config.noise_model == NoiseModel.MIXED:
            # Gaussian + Poisson
            gauss = torch.randn(shape, device=self.device) * noise_std
            poisson = torch.poisson(torch.ones(shape, device=self.device)) - 1.0
            return gauss + poisson
        
        elif self.config.noise_model == NoiseModel.QUANTIZED:
            # Quantization noise
            max_val = 2 ** self.config.quantization_bits
            return torch.round(torch.randn(shape, device=self.device) * noise_std) / max_val
        
        else:
            return torch.zeros(shape, device=self.device)
    
    def measure(self, x: torch.Tensor, pattern_type: Optional[PatternType] = None,
                num_patterns: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform SPI measurements with generated patterns.
        
        Args:
            x: Image (batch, num_pixels) or (num_pixels,)
            pattern_type: Type of patterns (if None, uses config)
            num_patterns: Number of patterns (if None, uses config)
        
        Returns:
            (measurements, patterns): measurement vector and pattern matrix
        """
        if pattern_type is None:
            pattern_type = self.config.pattern_type
        if num_patterns is None:
            num_patterns = self.config.num_measurements
        
        # Generate patterns
        if pattern_type == PatternType.RANDOM_BERNOULLI:
            h = self.pattern_gen.generate_random_bernoulli(num_patterns)
        elif pattern_type == PatternType.HADAMARD:
            h = self.pattern_gen.generate_hadamard(num_patterns)
        elif pattern_type == PatternType.GAUSSIAN:
            h = self.pattern_gen.generate_gaussian(num_patterns)
        elif pattern_type == PatternType.STRUCTURED:
            h = self.pattern_gen.generate_structured()
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Compute measurements
        y = self.forward(x, h)
        
        # Store history
        if x.dim() == 1:
            self.ground_truth = x.clone()
        self.measurements.extend(y.cpu().numpy())
        self.patterns.append(h)
        
        return y, h
    
    def get_measurement_matrix(self) -> torch.Tensor:
        """Get stacked measurement matrix H_N (all patterns as rows)."""
        if len(self.patterns) == 0:
            return torch.empty((0, self.config.num_pixels), device=self.device)
        
        return torch.cat(self.patterns, dim=0)
    
    def get_measurements_vector(self) -> torch.Tensor:
        """Get stacked measurement vector y_N."""
        return torch.tensor(self.measurements, device=self.device)


class DisorderAveragedOperator:
    """
    Estimate effective SPI operator under disorder (Section 2.4 of TPAMI).
    
    Models pattern disorder: h_n = h_eff + Δh_n
    
    Produces:
      - Aeff: Effective operator E[A]
      - Σmodel: Disorder-induced covariance
    """
    
    def __init__(self, config: SPIConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        self.aeff = None
        self.sigma_model = None
    
    def estimate_from_patterns(self, patterns: List[torch.Tensor],
                               prior_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate effective operator from pattern ensemble.
        
        Phase 1 of Algorithm 2 (TPAMI):
          Aeff = (1/K) Σ A_k
          Σmodel = Cov{(A_k - Aeff) z}
        
        Args:
            patterns: List of K pattern matrices {h_k}, each (M,)
            prior_samples: Probe signals (J, M) ~ p_prior(x)
        
        Returns:
            (aeff, sigma_model): Effective operator and disorder covariance
        """
        K = len(patterns)
        M = self.config.num_pixels
        J = prior_samples.shape[0]
        
        patterns_tensor = torch.stack(patterns)  # (K, M)
        prior_samples = prior_samples.to(self.device)
        
        # Compute effective operator
        self.aeff = torch.mean(patterns_tensor, dim=0)  # (M,)
        
        # Compute residuals for each disorder realization
        residuals = []
        for k in range(K):
            delta_h = patterns_tensor[k] - self.aeff  # (M,)
            # Residual from each probe signal
            residual_k = (delta_h.unsqueeze(0) * prior_samples).mean(dim=0)  # (M,)
            residuals.append(residual_k)
        
        residuals_tensor = torch.stack(residuals)  # (K, M)
        
        # Empirical covariance of residuals
        residuals_centered = residuals_tensor - residuals_tensor.mean(dim=0, keepdim=True)
        self.sigma_model = (residuals_centered.T @ residuals_centered) / (K - 1)  # (M, M)
        
        return self.aeff, self.sigma_model
    
    def estimate_from_bernoulli_disorder(self, 
                                        p_disorder: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic estimation for Bernoulli pattern disorder.
        
        If each bit flips with probability p_disorder, compute effective
        statistics analytically.
        
        Args:
            p_disorder: Probability of each bit flip
        
        Returns:
            (aeff, sigma_model): Effective operator and covariance
        """
        M = self.config.num_pixels
        
        # Effective pattern: h_eff[i] = p (bias toward 1)
        # For symmetric Bernoulli: h_eff[i] = 0.5
        self.aeff = torch.ones(M, device=self.device) * 0.5
        
        # Covariance of (h - h_eff): Cov[h_i] = p(1-p)
        var_per_bit = p_disorder * (1 - p_disorder)
        
        # Assuming independence between bits
        self.sigma_model = torch.eye(M, device=self.device) * var_per_bit
        
        return self.aeff, self.sigma_model


class SPISequentialMeasurement:
    """
    Sequential measurement protocol for SPI with optional adaptive design.
    
    Accumulates measurements and optionally optimizes next pattern based on
    information gain or other criteria (OED - optimal experimental design).
    """
    
    def __init__(self, forward_model: SPIForwardModel, device: str = "cpu"):
        self.forward_model = forward_model
        self.device = device
        
        self.measurement_count = 0
        self.stopping_time = None
        self.measurement_log = []
    
    def add_measurement(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> float:
        """
        Take a single SPI measurement.
        
        Args:
            x: Image signal
            h: Measurement pattern (if None, generates new)
        
        Returns:
            Scalar measurement y
        """
        if h is None:
            h = self.forward_model.pattern_gen.generate_random_bernoulli(num_patterns=1)
        
        y = self.forward_model.forward(x, h).item()
        
        self.measurement_log.append({
            'count': self.measurement_count,
            'measurement': y,
            'pattern': h.clone()
        })
        self.measurement_count += 1
        
        return y
    
    def adaptive_next_pattern(self, x_current: torch.Tensor,
                             information_gain_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        Generate next pattern via OED (optimal experimental design).
        
        Maximizes information gain: I(x; y | y_{1:n-1})
        
        Args:
            x_current: Current reconstruction estimate
            information_gain_fn: Function computing I(h, x_current)
        
        Returns:
            Optimized pattern
        """
        # Generate candidates
        candidates = self.forward_model.pattern_gen.generate_random_bernoulli(num_patterns=20)
        
        if information_gain_fn is None:
            # Default: entropy of measurement prediction
            def information_gain_fn(h, x):
                pred = self.forward_model.forward(x, h)
                # Higher variance = higher information
                return torch.abs(pred - 0.5)  # Distance from uniform
        
        # Evaluate and select best
        gains = []
        for h in candidates:
            gain = information_gain_fn(h, x_current)
            gains.append(gain.item() if torch.is_tensor(gain) else gain)
        
        best_idx = np.argmax(gains)
        return candidates[best_idx:best_idx+1]
    
    def get_summary(self) -> Dict:
        """Get summary of sequential measurements."""
        return {
            'total_measurements': self.measurement_count,
            'stopping_time': self.stopping_time,
            'measurements': [log['measurement'] for log in self.measurement_log],
            'num_patterns': len([log['pattern'] for log in self.measurement_log])
        }


def main_example():
    """Example demonstrating SPI forward model and measurement."""
    
    print("=" * 70)
    print("Single-Pixel Imaging (SPI) Forward Model")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration
    config = SPIConfig(
        image_dim=32,
        num_measurements=256,
        pattern_type=PatternType.RANDOM_BERNOULLI,
        noise_model=NoiseModel.GAUSSIAN,
        measurement_noise_std=0.1
    )
    
    print(f"\nConfiguration:")
    print(f"  Image size: {config.image_dim}×{config.image_dim} = {config.num_pixels} pixels")
    print(f"  Measurements: {config.num_measurements}")
    print(f"  Pattern type: {config.pattern_type.value}")
    print(f"  Noise model: {config.noise_model.value}")
    print(f"  Noise std: {config.measurement_noise_std}\n")
    
    # Create forward model
    forward_model = SPIForwardModel(config, device=device)
    
    # Create synthetic image
    print("1. Creating synthetic image")
    print("-" * 70)
    x_true = torch.rand(config.num_pixels, device=device) * 0.8 + 0.1  # [0.1, 0.9]
    print(f"  Image statistics:")
    print(f"    Min: {x_true.min():.4f}, Max: {x_true.max():.4f}")
    print(f"    Mean: {x_true.mean():.4f}, Std: {x_true.std():.4f}")
    
    # Take measurements
    print("\n2. Taking SPI measurements")
    print("-" * 70)
    
    for pattern_type in [PatternType.RANDOM_BERNOULLI, PatternType.HADAMARD, PatternType.GAUSSIAN]:
        config.pattern_type = pattern_type
        y, h = forward_model.measure(x_true, num_patterns=50)
        
        print(f"  {pattern_type.value.upper()}:")
        print(f"    Measurement shape: {y.shape}")
        print(f"    Measurement range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"    Measurement SNR: {(y.std() / config.measurement_noise_std):.2f}")
    
    # Disorder-averaging
    print("\n3. Disorder-Averaged Operator Estimation")
    print("-" * 70)
    
    config.pattern_type = PatternType.RANDOM_BERNOULLI
    
    # Generate pattern ensemble
    patterns = []
    for k in range(50):
        h_k = torch.randint(0, 2, (config.num_pixels,), dtype=torch.float32, device=device)
        patterns.append(h_k)
    
    # Estimate effective operator
    disorder_est = DisorderAveragedOperator(config, device=device)
    prior_samples = torch.rand(16, config.num_pixels, device=device)
    
    aeff, sigma_model = disorder_est.estimate_from_patterns(patterns, prior_samples)
    
    print(f"  Effective operator Aeff:")
    print(f"    Shape: {aeff.shape}")
    print(f"    Mean: {aeff.mean():.4f}, Std: {aeff.std():.4f}")
    print(f"  Disorder covariance Σmodel:")
    print(f"    Shape: {sigma_model.shape}")
    print(f"    Trace: {sigma_model.trace():.4f}")
    print(f"    Frobenius norm: {torch.norm(sigma_model):.4f}")
    
    # Sequential measurement
    print("\n4. Sequential Measurement")
    print("-" * 70)
    
    seq_measure = SPISequentialMeasurement(forward_model, device=device)
    
    for n in range(10):
        y = seq_measure.add_measurement(x_true)
        print(f"  Measurement {n+1:2d}: y = {y:.6f}")
    
    summary = seq_measure.get_summary()
    print(f"\n  Total measurements taken: {summary['total_measurements']}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main_example()
