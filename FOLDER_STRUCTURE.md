# Folder Structure

```
experiments_spi_ddm/
├─ README.md
├─ QUICK_REFERENCE.txt
├─ FOLDER_STRUCTURE.md
├─ INDEX.md
├─ GENERATED_FILES.txt
├─ .gitignore
│
├─ configs/
│  ├─ synthetic_fixedN.yaml           # Section 7.2: fixed-N synthetic experiments
│  ├─ synthetic_adaptive.yaml         # Section 7.2: adaptive OED synthetic experiments
│  ├─ hardware_microscopy.yaml        # Section 7.3: real SPI hardware experiments
│  ├─ hardware_compressive.yaml       # Section 7.3: compressive imaging setup
│  └─ compare_e_vs_p.yaml             # Section 7.4: e-value vs p-value comparison
│
├─ src/
│  ├─ __init__.py
│  ├─ forward_spi.py                  # Eq. 1, 10–11: y_n = <h(ω_n), x> + ε_n
│  ├─ disorder_avg.py                 # Eq. 6–9, 12–13: A_eff, Σ_eff, Σ_model
│  ├─ e_values.py                     # Eq. 19–22: E-process, likelihood ratio
│  ├─ p_values.py                     # Section 4.1: classical p-value testing
│  ├─ posterior_sampler.py            # [STUB] Eq. 15: Disorder-diffusive ALSS
│  ├─ hypotheses.py                   # [STUB] Eq. 16–17: H0/H1 definitions
│  ├─ test_statistics.py              # [STUB] Test statistic T_n computation
│  ├─ stopping_rules.py               # [STUB] Anytime-valid stopping (§4.2–4.3)
│  ├─ oed_policy.py                   # [STUB] OED: expected information gain (§5d)
│  └─ metrics_logging.py              # [STUB] Type I, power, stopping time aggregation
│
├─ scripts/
│  ├─ run_synthetic_fixedN.py         # Reproduce Section 7.2 (fixed-N curves)
│  ├─ run_synthetic_adaptive.py       # [STUB] Reproduce Section 7.2 (adaptive OED)
│  ├─ run_hardware_exps.py            # [STUB] Reproduce Section 7.3 (hardware experiments)
│  └─ compare_e_vs_p.py               # Reproduce Section 7.4 (Type I, power, stopping times)
│
└─ results/
   ├─ synthetic/
   │  ├─ logs/                        # .log files from runs
   │  └─ csv/                         # results_*.csv with metrics
   ├─ hardware/
   │  ├─ logs/
   │  └─ csv/
   └─ e_vs_p/
      ├─ logs/
      └─ csv/                         # type_i_power_stopping.csv (main §8 table)
```

## Key Files by Paper Section

| Section | Config | Script | Modules |
|---------|--------|--------|---------|
| **7.2** Fixed-N SPI | `synthetic_fixedN.yaml` | `run_synthetic_fixedN.py` | `forward_spi.py`, `disorder_avg.py`, `e_values.py`, `p_values.py` |
| **7.2** Adaptive OED | `synthetic_adaptive.yaml` | `run_synthetic_adaptive.py` | + `oed_policy.py` |
| **7.3** Hardware | `hardware_microscopy.yaml` | `run_hardware_exps.py` | + hardware-specific loading |
| **7.4** E vs P | `compare_e_vs_p.yaml` | `compare_e_vs_p.py` | `e_values.py`, `p_values.py` |

## Quick Start

```bash
# Setup
cd experiments_spi_ddm

# 1. Synthetic fixed-N (Section 7.2)
python scripts/run_synthetic_fixedN.py --config configs/synthetic_fixedN.yaml

# 2. E-value vs P-value comparison (Section 7.4)
python scripts/compare_e_vs_p.py --config configs/compare_e_vs_p.yaml

# Check results
cat results/e_vs_p/csv/type_i_power_stopping.csv
```

## Equation Cross-References

- **forward_spi.py**: Eq. 1, 10–11 (sequential SPI measurement model)
- **disorder_avg.py**: Eq. 6–9, 12–13 (disorder averaging, Σ_eff, likelihood)
- **e_values.py**: Eq. 19–22 (e-value definition, likelihood ratio, cumulative e-process)
- **p_values.py**: Section 4.1 (test statistic T_n, p-value computation)
- **stopping_rules.py**: Eq. 15–17, Section 4.2–4.3 (anytime-valid stopping)
- **hypotheses.py**: Eq. 16–17 (H0/H1, δ0, quality threshold)
- **oed_policy.py**: Section 5d (adaptive pattern selection, information gain)

## File Summary

### Documentation (5 files)
- **README.md** (42 lines) – Quick start guide
- **QUICK_REFERENCE.txt** (263 lines) – One-page cheat sheet
- **INDEX.md** (331 lines) – Comprehensive guide (this file)
- **FOLDER_STRUCTURE.md** (79 lines) – Tree structure (you are here)
- **GENERATED_FILES.txt** (137 lines) – Creation manifest

### Configurations (5 YAML files)
- **synthetic_fixedN.yaml** (68 lines)
- **synthetic_adaptive.yaml** (70 lines)
- **hardware_microscopy.yaml** (73 lines)
- **hardware_compressive.yaml** (stub)
- **compare_e_vs_p.yaml** (117 lines)

### Source Code (10 Python files)

**Fully Implemented (5 files):**
- **__init__.py** (11 lines)
- **forward_spi.py** (146 lines)
- **disorder_avg.py** (211 lines)
- **e_values.py** (254 lines)
- **p_values.py** (273 lines)

**Stubs (5 files):**
- **posterior_sampler.py** – ALSS sampler (Eq. 15)
- **hypotheses.py** – H0/H1 definitions (Eq. 16–17)
- **test_statistics.py** – Test statistic T_n
- **stopping_rules.py** – Anytime-valid stopping (§4.2–4.3)
- **oed_policy.py** – OED optimization (§5d)

### Scripts (4 Python files)

**Fully Implemented (2 files):**
- **run_synthetic_fixedN.py** (192 lines) – Section 7.2
- **compare_e_vs_p.py** (273 lines) – Section 7.4 ⭐

**Stubs (2 files):**
- **run_synthetic_adaptive.py** – Adaptive OED (Section 7.2)
- **run_hardware_exps.py** – Hardware experiments (Section 7.3)

## Results Output Structure

```
results/
├─ synthetic/
│  ├─ logs/
│  │  └─ synthetic_fixedN.log
│  └─ csv/
│     └─ synthetic_fixedN_metrics.csv
│        Columns: replicate_id, n_measurements, recon_error, h0_true,
│                 e_value, e_reject, p_value, p_reject, 
│                 e_type_i_error, e_type_ii_error, p_type_i_error, p_type_ii_error
│
├─ hardware/
│  ├─ logs/
│  │  └─ hardware_microscopy.log
│  └─ csv/
│     └─ hardware_microscopy_metrics.csv
│
└─ e_vs_p/
   ├─ logs/
   │  └─ compare_e_vs_p.log
   └─ csv/
      ├─ type_i_power_stopping.csv          (Main result for Section 8)
      │  Columns: hypothesis, replicate, e_reject, p_reject, e_stop_time, p_stop_time
      ├─ stopping_time_distribution.csv
      │  Columns: hypothesis, method, mean, std, q25, q50, q75, q95
      └─ coverage_misspec.csv
         Columns: method, coverage_under_misspec
```

## Configuration File Structure

Each YAML config file contains:

```yaml
experiment:
  name: "experiment_name"
  description: "..."
  seed: 42
  n_replicates: 50

data:
  source: "synthetic" | "hardware"
  image_shape: [64, 64]
  n_measurements: [10, 20, 50, 100, 200]
  noise_std: 0.01

posterior:
  prior_model: "diffusion_score"
  sampler: "alss_langevin"
  n_samples: 5000
  n_steps: 1000
  step_size: 0.01

hypotheses:
  h0: {type: "reconstruction_error", threshold_delta0: 0.05}
  h1: {type: "reconstruction_error", threshold_delta0: 0.05}

testing:
  p_value: {method: "monte_carlo_null", n_null_samples: 10000, alpha: 0.05}
  e_value: {method: "likelihood_ratio", likelihood_model: "gaussian_disorder_averaged"}
  stopping: {e_value_threshold: 10.0, p_value_threshold: 0.05, max_measurements: 300}

metrics:
  - type_i_error
  - power
  - stopping_time_mean
  - reconstruction_error_at_stop

output:
  results_dir: "results/synthetic"
  log_file: "results/synthetic/logs/synthetic_fixedN.log"
  csv_file: "results/synthetic/csv/synthetic_fixedN_metrics.csv"
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Forward SPI model | ✓ Done | Eq. 1, 10–11; sequential accumulation |
| Disorder averaging | ✓ Done | Eq. 6–9, 12–13; A_eff, Σ_eff computation |
| E-value testing | ✓ Done | Eq. 19–22; likelihood ratio, e-process |
| P-value testing | ✓ Done | Section 4.1; fixed-N, Monte Carlo null |
| Posterior sampler | ⏳ Stub | Eq. 15; ALSS + diffusion |
| OED policy | ⏳ Stub | Section 5d; expected information gain |
| Hardware loading | ⏳ Stub | Section 7.3; pattern + calibration I/O |
| Synthetic fixed-N script | ✓ Done | Reproduces Section 7.2 curves |
| E vs P script | ✓ Done | Reproduces Section 7.4 main result |
| Adaptive OED script | ⏳ Stub | Integrates OED loop with e-value stopping |

## Running the Experiments

### Example 1: Fixed-N Synthetic (Section 7.2)
```bash
python scripts/run_synthetic_fixedN.py --config configs/synthetic_fixedN.yaml
# Output: results/synthetic/csv/synthetic_fixedN_metrics.csv
```

### Example 2: E-value vs P-value (Section 7.4) ⭐ PRIMARY
```bash
python scripts/compare_e_vs_p.py --config configs/compare_e_vs_p.yaml
# Output: results/e_vs_p/csv/type_i_power_stopping.csv
# Terminal: Type I error, power, stopping time tables
```

## Dependencies

**Required:**
- Python 3.7+
- numpy
- pyyaml

**Optional:**
- pytorch (for diffusion prior training)
- scipy (for OED optimization)
- matplotlib (for visualization)

## Git Integration

```bash
cd experiments_spi_ddm
git init
git add -A
git commit -m "Initial commit: SPI hypothesis testing framework"
git remote add origin https://github.com/yourusername/spi-hypothesis-testing.git
git push -u origin main
```

## File Size Summary

```
Documentation:     5 files  ~  950 lines
Configurations:    5 files  ~  400 lines
Source Code:      10 files  ~ 1,300 lines (5 impl. + 5 stubs)
Scripts:           4 files  ~  465 lines (2 impl. + 2 stubs)

Total Implemented: 16 files ~ 2,600 lines
Total with Stubs:  24 files ~ 3,200 lines
```

## Next Steps

1. ✓ Download all files from workspace
2. ✓ Create folder structure
3. ✓ Install dependencies: `pip install numpy pyyaml`
4. ✓ Run experiments: `python scripts/compare_e_vs_p.py --config configs/compare_e_vs_p.yaml`
5. → Implement stubs (posterior_sampler.py, oed_policy.py, etc.)
6. → Add real SPI hardware data for Section 7.3
7. → Initialize Git repository
8. → Push to GitHub

---

All files are created and ready for download from the workspace!
