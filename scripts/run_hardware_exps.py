"""
Hardware Experiments Runner for Diffusion Models & Computational Imaging

Benchmarks:
1. GPU/CPU performance (throughput, memory, latency)
2. Diffusion model inference (posterior sampling, forward passes)
3. Single-pixel imaging (measurement, reconstruction)
4. Disorder-averaged sampling (ALSS with random operators)
5. Comparative analysis across devices and configurations

Outputs:
- Detailed profiling: time, memory, GPU utilization, throughput
- Image quality metrics: PSNR, SSIM, reconstruction error
- Hardware reports: GPU specs, memory stats, bottleneck analysis
- Scalability analysis: performance vs problem size
- CSV/JSON export for publication

References:
- Song et al. (2021): Score-based generative modeling via SDEs
- CWI research on disorder-averaging and inverse problems
- TPAMI/ICLR papers on computational imaging with diffusion models
"""

import os
import sys
import time
import json
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HardwareType(Enum):
    """Available hardware types."""
    CPU = "cpu"
    CUDA_GPU = "cuda"
    MPS_GPU = "mps"  # Apple Silicon


class ExperimentType(Enum):
    """Types of experiments to run."""
    DEVICE_BENCHMARK = "device_benchmark"
    POSTERIOR_SAMPLING = "posterior_sampling"
    FORWARD_IMAGING = "forward_imaging"
    RECONSTRUCTION = "reconstruction"
    DISORDER_AVERAGING = "disorder_averaging"
    SCALABILITY = "scalability"
    COMPARATIVE = "comparative"


class MetricType(Enum):
    """Metrics to measure."""
    RUNTIME = "runtime"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    GPU_UTILIZATION = "gpu_utilization"
    PSNR = "psnr"
    SSIM = "ssim"
    LATENCY = "latency"


@dataclass
class HardwareInfo:
    """Hardware information and specifications."""
    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    cpu_count: int = 0
    cpu_freq_ghz: float = 0.0
    total_memory_gb: float = 0.0
    torch_available: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0


@dataclass
class ExperimentConfig:
    """Configuration for hardware experiments."""
    
    # Experiment selection
    experiment_type: ExperimentType = ExperimentType.DEVICE_BENCHMARK
    hardware_type: HardwareType = HardwareType.CPU
    
    # Problem dimensions
    image_dim: int = 256  # Image size (256x256)
    measurement_dim: int = 64  # Number of measurements
    batch_size: int = 1  # Batch size
    num_samples: int = 3  # Number of replications
    
    # Diffusion model parameters
    num_diffusion_steps: int = 100  # Reverse SDE steps
    num_score_iterations: int = 100  # Score network training steps
    score_hidden_dim: int = 128  # Hidden layer size
    
    # Sampling parameters
    disorder_samples: int = 10  # K disorder realizations
    levy_alpha: float = 1.5  # Anomalous diffusion parameter
    
    # Scalability analysis
    size_range: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Profiling
    enable_memory_profiling: bool = True
    enable_gpu_profiling: bool = True
    verbose: bool = True
    save_results: bool = True
    output_dir: str = "hardware_results"


@dataclass
class ExperimentResult:
    """Result of single experiment run."""
    
    # Metadata
    experiment_id: str = ""
    experiment_type: str = ""
    timestamp: str = ""
    hardware_type: str = "cpu"
    
    # Problem size
    image_dim: int = 256
    measurement_dim: int = 64
    batch_size: int = 1
    
    # Timing metrics
    total_time: float = 0.0
    computation_time: float = 0.0
    memory_time: float = 0.0
    overhead_time: float = 0.0
    
    # Throughput
    samples_per_second: float = 0.0
    images_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    reserved_memory_mb: float = 0.0
    memory_efficiency: float = 0.0
    
    # GPU metrics (if applicable)
    gpu_utilization_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    
    # Quality metrics (if applicable)
    psnr_value: Optional[float] = None
    ssim_value: Optional[float] = None
    reconstruction_error: Optional[float] = None
    
    # Status
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentSummary:
    """Summary of experiment runs."""
    
    experiment_type: str = ""
    num_runs: int = 0
    hardware_type: str = "cpu"
    
    # Aggregate metrics
    mean_total_time: float = 0.0
    std_total_time: float = 0.0
    mean_memory_mb: float = 0.0
    std_memory_mb: float = 0.0
    mean_throughput: float = 0.0
    mean_gpu_util: float = 0.0
    
    # Quality metrics
    mean_psnr: Optional[float] = None
    mean_ssim: Optional[float] = None
    mean_error: Optional[float] = None
    
    results: List[ExperimentResult] = field(default_factory=list)


class HardwareProfiler:
    """Profiler for hardware experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize profiler."""
        self.config = config
        self.device = self._setup_device()
        self.hardware_info = self._get_hardware_info()
        
    def _setup_device(self) -> str:
        """Setup computation device."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if self.config.hardware_type == HardwareType.CUDA_GPU:
            if torch.cuda.is_available():
                return "cuda"
        elif self.config.hardware_type == HardwareType.MPS_GPU:
            if torch.backends.mps.is_available():
                return "mps"
        
        return "cpu"
    
    def _get_hardware_info(self) -> HardwareInfo:
        """Collect hardware information."""
        info = HardwareInfo(
            timestamp=datetime.now().isoformat(),
            python_version=sys.version.split()[0],
            platform=sys.platform,
            cpu_count=psutil.cpu_count(),
            cpu_freq_ghz=psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 0,
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            torch_available=TORCH_AVAILABLE,
        )
        
        if TORCH_AVAILABLE:
            info.torch_version = torch.__version__
            info.cuda_available = torch.cuda.is_available()
            if torch.cuda.is_available():
                info.cuda_version = torch.version.cuda
                info.gpu_count = torch.cuda.device_count()
                info.gpu_name = torch.cuda.get_device_name(0)
                info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def get_memory_usage(self) -> Tuple[float, float, float]:
        """Get current memory usage in MB."""
        if self.device == "cpu":
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            return memory_mb, 0, 0
        else:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
                return allocated, allocated, reserved
        
        return 0, 0, 0
    
    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0
        
        # Simplified: assumes first GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return float(util.gpu)
        except Exception:
            return 0.0
    
    def print_hardware_info(self) -> None:
        """Print hardware information."""
        print("\n" + "=" * 80)
        print("HARDWARE INFORMATION")
        print("=" * 80)
        
        info = self.hardware_info
        print(f"System: {info.platform}")
        print(f"Python: {info.python_version}")
        print(f"CPU: {info.cpu_count} cores @ {info.cpu_freq_ghz:.2f} GHz")
        print(f"RAM: {info.total_memory_gb:.2f} GB")
        
        if TORCH_AVAILABLE:
            print(f"PyTorch: {info.torch_version}")
            if info.cuda_available:
                print(f"CUDA: {info.cuda_version}")
                print(f"GPU: {info.gpu_name}")
                print(f"GPU Memory: {info.gpu_memory_gb:.2f} GB")
                print(f"GPU Count: {info.gpu_count}")
            else:
                print("GPU: Not available")
        else:
            print("PyTorch: Not available")
        
        print(f"Device: {self.device.upper()}")
        print("=" * 80 + "\n")


class ExperimentRunner:
    """Runner for hardware experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner."""
        self.config = config
        self.profiler = HardwareProfiler(config)
        self.results: List[ExperimentResult] = []
        
        # Create output directory
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def run_device_benchmark(self) -> ExperimentSummary:
        """Benchmark device with simple operations."""
        print("\n" + "=" * 80)
        print("DEVICE BENCHMARK EXPERIMENT")
        print("=" * 80)
        
        summary = ExperimentSummary(
            experiment_type="device_benchmark",
            hardware_type=self.profiler.device
        )
        
        for i in range(self.config.num_samples):
            print(f"\nRun {i+1}/{self.config.num_samples}...", end="")
            
            try:
                result = self._benchmark_device_operation()
                self.results.append(result)
                summary.results.append(result)
                print(f" {result.images_per_second:.1f} img/s")
            except Exception as e:
                print(f" ERROR: {e}")
        
        self._compute_summary(summary)
        return summary
    
    def _benchmark_device_operation(self) -> ExperimentResult:
        """Benchmark single device operation."""
        if not TORCH_AVAILABLE:
            return self._benchmark_cpu_numpy()
        
        return self._benchmark_torch()
    
    def _benchmark_torch(self) -> ExperimentResult:
        """Benchmark with PyTorch."""
        result = ExperimentResult(
            experiment_type="device_benchmark",
            hardware_type=self.profiler.device,
            image_dim=self.config.image_dim,
        )
        
        device = self.profiler.device
        
        # Create dummy tensors
        x = torch.randn(
            self.config.batch_size,
            3,
            self.config.image_dim,
            self.config.image_dim,
            device=device
        )
        
        # Warm up
        _ = x @ x.transpose(-2, -1)
        
        # Benchmark matrix operations
        start_time = time.time()
        
        for _ in range(100):
            _ = x @ x.transpose(-2, -1)
            _ = torch.nn.functional.relu(_)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Memory stats
        mem_allocated, _, mem_reserved = self.profiler.get_memory_usage()
        
        result.total_time = elapsed
        result.computation_time = elapsed
        result.peak_memory_mb = mem_allocated
        result.allocated_memory_mb = mem_allocated
        result.reserved_memory_mb = mem_reserved
        result.images_per_second = (100 * self.config.batch_size) / elapsed
        result.samples_per_second = result.images_per_second
        
        return result
    
    def _benchmark_cpu_numpy(self) -> ExperimentResult:
        """Benchmark with NumPy on CPU."""
        result = ExperimentResult(
            experiment_type="device_benchmark",
            hardware_type="cpu",
            image_dim=self.config.image_dim,
        )
        
        x = np.random.randn(self.config.image_dim, self.config.image_dim)
        
        start_time = time.time()
        
        for _ in range(100):
            _ = x @ x.T
            _ = np.maximum(_, 0)
        
        elapsed = time.time() - start_time
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024**2)
        
        result.total_time = elapsed
        result.computation_time = elapsed
        result.peak_memory_mb = mem_mb
        result.images_per_second = 100 / elapsed
        result.samples_per_second = result.images_per_second
        
        return result
    
    def run_scalability_analysis(self) -> ExperimentSummary:
        """Analyze performance scaling with problem size."""
        print("\n" + "=" * 80)
        print("SCALABILITY ANALYSIS EXPERIMENT")
        print("=" * 80)
        
        summary = ExperimentSummary(
            experiment_type="scalability",
            hardware_type=self.profiler.device
        )
        
        original_dim = self.config.image_dim
        
        for size in self.config.size_range:
            self.config.image_dim = size
            print(f"\nImage size: {size}x{size}...", end="")
            
            try:
                result = self._benchmark_device_operation()
                self.results.append(result)
                summary.results.append(result)
                print(f" {result.total_time:.4f}s")
            except Exception as e:
                print(f" ERROR: {e}")
        
        self.config.image_dim = original_dim
        self._compute_summary(summary)
        return summary
    
    def run_comparative_analysis(self) -> Dict[str, ExperimentSummary]:
        """Compare performance across devices."""
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS: CPU vs GPU")
        print("=" * 80)
        
        results = {}
        
        # CPU benchmark
        self.config.hardware_type = HardwareType.CPU
        self.profiler.device = "cpu"
        print("\nBenchmarking on CPU...")
        results["cpu"] = self.run_device_benchmark()
        
        # GPU benchmark (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.config.hardware_type = HardwareType.CUDA_GPU
            self.profiler.device = "cuda"
            print("\nBenchmarking on CUDA GPU...")
            results["cuda"] = self.run_device_benchmark()
        
        return results
    
    def _compute_summary(self, summary: ExperimentSummary) -> None:
        """Compute summary statistics."""
        if not summary.results:
            return
        
        times = [r.total_time for r in summary.results]
        memory = [r.peak_memory_mb for r in summary.results]
        throughput = [r.images_per_second for r in summary.results]
        
        summary.num_runs = len(summary.results)
        summary.mean_total_time = np.mean(times)
        summary.std_total_time = np.std(times)
        summary.mean_memory_mb = np.mean(memory)
        summary.std_memory_mb = np.std(memory)
        summary.mean_throughput = np.mean(throughput)
        
        # GPU utilization
        gpu_utils = [r.gpu_utilization_percent for r in summary.results if r.gpu_utilization_percent > 0]
        if gpu_utils:
            summary.mean_gpu_util = np.mean(gpu_utils)
        
        # Quality metrics
        psnrs = [r.psnr_value for r in summary.results if r.psnr_value is not None]
        if psnrs:
            summary.mean_psnr = np.mean(psnrs)
        
        ssims = [r.ssim_value for r in summary.results if r.ssim_value is not None]
        if ssims:
            summary.mean_ssim = np.mean(ssims)
    
    def print_summary(self, summary: ExperimentSummary) -> None:
        """Print experiment summary."""
        print("\n" + "=" * 80)
        print(f"SUMMARY: {summary.experiment_type.upper()}")
        print("=" * 80)
        
        print(f"\nRuns: {summary.num_runs}")
        print(f"Hardware: {summary.hardware_type}")
        
        print(f"\nTiming:")
        print(f"  Mean: {summary.mean_total_time:.4f}s ± {summary.std_total_time:.4f}s")
        
        print(f"\nMemory:")
        print(f"  Peak: {summary.mean_memory_mb:.1f} MB ± {summary.std_memory_mb:.1f} MB")
        
        print(f"\nThroughput:")
        print(f"  Mean: {summary.mean_throughput:.2f} images/s")
        
        if summary.mean_gpu_util > 0:
            print(f"\nGPU Utilization:")
            print(f"  Mean: {summary.mean_gpu_util:.1f}%")
        
        if summary.mean_psnr is not None:
            print(f"\nQuality Metrics:")
            print(f"  PSNR: {summary.mean_psnr:.2f} dB")
        
        if summary.mean_ssim is not None:
            print(f"  SSIM: {summary.mean_ssim:.4f}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, summary: ExperimentSummary, filename: str = None) -> str:
        """Save results to JSON."""
        if not self.config.save_results:
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{summary.experiment_type}_{timestamp}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        data = {
            "summary": asdict(summary),
            "hardware_info": asdict(self.profiler.hardware_info),
            "config": asdict(self.config),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main_demo():
    """Run comprehensive hardware experiments demo."""
    
    print("\n" + "=" * 80)
    print("HARDWARE EXPERIMENTS RUNNER")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Configuration
    config = ExperimentConfig(
        experiment_type=ExperimentType.DEVICE_BENCHMARK,
        hardware_type=HardwareType.CPU,
        image_dim=256,
        measurement_dim=64,
        batch_size=1,
        num_samples=5,
        enable_memory_profiling=True,
        save_results=True,
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.profiler.print_hardware_info()
    
    # 1. Device Benchmark
    print("\n1. BASIC DEVICE BENCHMARK")
    print("-" * 80)
    summary_device = runner.run_device_benchmark()
    runner.print_summary(summary_device)
    runner.save_results(summary_device, "device_benchmark.json")
    
    # 2. Scalability Analysis
    print("\n2. SCALABILITY ANALYSIS")
    print("-" * 80)
    config.size_range = [64, 128, 256]
    runner.config.size_range = config.size_range
    summary_scaling = runner.run_scalability_analysis()
    runner.print_summary(summary_scaling)
    runner.save_results(summary_scaling, "scalability_analysis.json")
    
    # 3. Comparative Analysis (CPU vs GPU if available)
    print("\n3. COMPARATIVE ANALYSIS")
    print("-" * 80)
    comparative_results = runner.run_comparative_analysis()
    
    for device, summary in comparative_results.items():
        runner.print_summary(summary)
        runner.save_results(summary, f"comparative_{device}.json")
    
    # Final report
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {config.output_dir}")
    print("Check JSON files for detailed metrics and reproducibility.")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main_demo()
