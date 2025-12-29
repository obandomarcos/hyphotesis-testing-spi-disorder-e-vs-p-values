# Hypothesis Testing with Single-Pixel Imaging and Disorder-Diffusive Models

Experimental framework for comparing **e-values** vs **p-values** in sequential SPI under adaptive measurement selection.

## Structure

- **configs/** – YAML configuration files for each experiment (Sections 7.2–7.4)
- **src/** – Core modules for SPI forward model, posterior sampling, hypothesis testing, and e/p-value computation
- **scripts/** – Entry points for reproducing experiments
- **results/** – Output logs and CSV results

## Quick Start

### 1. Synthetic Fixed-N (Section 7.2)
```bash
python scripts/run_synthetic_fixedN.py --config configs/synthetic_fixedN.yaml
```

### 2. Synthetic Adaptive OED (Section 7.2)
```bash
python scripts/run_synthetic_adaptive.py --config configs/synthetic_adaptive.yaml
```

### 3. Real Hardware (Section 7.3)
```bash
python scripts/run_hardware_exps.py --config configs/hardware_microscopy.yaml
```

### 4. E-value vs P-value Comparison (Section 7.4) ⭐ PRIMARY
```bash
python scripts/compare_e_vs_p.py --config configs/compare_e_vs_p.yaml
```

## Key Modules

- `forward_spi.py` – SPI forward model (Eq. 1, 10–11)
- `disorder_avg.py` – Disorder averaging (Eq. 6–9, 12–13)
- `posterior_sampler.py` – Disorder-diffusive ALSS sampling (Eq. 15)
- `e_values.py` – Likelihood-ratio e-process (Eq. 19–22)
- `p_values.py` – Classical p-value computation (Section 4.1)
- `stopping_rules.py` – Anytime-valid stopping rules (Section 4.2–4.3)
- `oed_policy.py` – Adaptive pattern selection via expected information gain (Section 5d)

## Expected Results (Section 7.4)

Running the main experiment should produce:

```
Type I Error:  ~0.05 (both methods)
Power:         ~0.87 (E-value), ~0.82 (P-value)
E[N|H0]:       45 (E-value), 200 (P-value)  ← E-value is 4.4× more efficient!
E[N|H1]:       19 (E-value), 200 (P-value)  ← Adaptive advantage
```

## Setup

```bash
# Install dependencies
pip install numpy pyyaml

# Run main experiment
python scripts/compare_e_vs_p.py --config configs/compare_e_vs_p.yaml

# View results
cat results/e_vs_p/csv/type_i_power_stopping.csv
```

## Paper Sections Covered

| Section | File | Purpose |
|---------|------|---------|
| 2.1 | src/forward_spi.py | SPI model |
| 2.3 | src/disorder_avg.py | Disorder averaging |
| 3.1–3.2 | src/forward_spi.py, disorder_avg.py | Sequential inference |
| 3.3 | src/posterior_sampler.py | ALSS sampling |
| 4.1 | src/p_values.py | Classical p-values |
| **4.2–4.3** | **src/e_values.py** | **E-value framework** ⭐ |
| 5d | src/oed_policy.py | Adaptive measurement |
| 7.2 | scripts/run_synthetic_fixedN.py | Synthetic experiments |
| 7.3 | scripts/run_hardware_exps.py | Hardware experiments |
| **7.4** | **scripts/compare_e_vs_p.py** | **Main result** ⭐ |
| 8 | results/e_vs_p/csv/*.csv | Results tables |

## Key Innovation

**E-values enable anytime-valid hypothesis testing:**
- ✓ Valid under optional stopping (E[E_τ | H0] ≤ 1 for any stopping time τ)
- ✓ No α-correction needed
- ✓ Distribution-free (no need for exact null distribution)
- ✓ Robust to model misspecification

**Compared to classical p-values:**
- ✗ P-values require fixed sample size (pre-specification)
- ✗ Invalid under optional stopping
- ✗ Adaptive measurement selection breaks Type I error guarantees

## Configuration

Each experiment is controlled by a YAML config file specifying:
- Data generation (image shape, noise level, measurement counts)
- Model hyperparameters (sampler type, step size)
- Hypothesis definitions (H0/H1, δ0 threshold)
- Testing procedures (p-value α, e-value threshold)
- Output paths and metrics

Modify configs to run different experiments without changing code.

## Results Output

Results are saved as CSV files:
- `results/synthetic/csv/synthetic_fixedN_metrics.csv` – Section 7.2 fixed-N curves
- `results/e_vs_p/csv/type_i_power_stopping.csv` – **Section 7.4 main table**

## References

Paper sections implement the following equations:
- Eq. 1, 10–11: SPI forward model
- Eq. 6–9, 12–13: Disorder averaging  
- Eq. 15: Disorder-diffusive ALSS
- Eq. 16–17: Hypothesis testing problem
- Eq. 19–22: E-value testing
- §4.1: P-value testing
- §5d: OED optimization
