# ML_3MLFIT — LLM-Guided Spectral Fitting of the Crab Nebula

Fits a log-parabola gamma-ray spectrum to HAWC Crab Nebula data using [3ML](https://threeml.readthedocs.io) and [HAWC HAL](https://github.com/threeML/hawc_hal). Includes a standard maximum-likelihood fit and an OPRO-style optimizer that uses a local LLM (via Ollama) to propose parameter updates.

## Project Structure

```
ML_3MLFIT/
├── scripts/
│   ├── crab.py                        # Standard 3ML/ROOT fit, saves results
│   ├── crab_opro.py                   # OPRO optimizer (early version)
│   ├── crab_opro_change_start_v6.py   # OPRO optimizer (current version)
│   ├── extract_crab_fit_training_data.py
│   └── crab_opro_change_start_v[1-5].py  # Previous iterations
├── data/
│   ├── HAWC_9bin_507days_crab_data.hd5
│   ├── HAWC_9bin_507days_crab_response.hd5
│   └── crab_lp_public_results.fits    # Output from standard fit
├── results/                           # Plots and logs (not tracked by git)
└── config/
    └── crab_fit.yml                   # Saved best-fit model
```

## Method

**Standard fit (`crab.py`):** Fits a log-parabola spectrum using 3ML's `JointLikelihood` with the ROOT minimizer over 9 HAWC energy bins (507 days of data).

**OPRO fit (`crab_opro_change_start_v6.py`):** Replaces the gradient-based minimizer with an LLM acting as the optimizer. At each iteration, the LLM receives the current best parameters, a finite-difference gradient estimate, and the top-10 evaluated points, then proposes the next parameter set to evaluate. Runs locally via Ollama (`llama3.1:8b`).

### Log-parabola spectral model

$$\frac{dN}{dE} = K \left(\frac{E}{E_\mathrm{piv}}\right)^{\alpha + \beta \ln(E/E_\mathrm{piv})}$$

| Parameter | Description | Bounds |
|-----------|-------------|--------|
| `K` | Flux normalization | (1e-44, 1e-19) keV⁻¹ cm⁻² s⁻¹ |
| `alpha` | Spectral index | (-4, 2) |
| `beta` | Spectral curvature | (-4, 2) |
| `piv` | Pivot energy (fixed) | 7 TeV |

## Dependencies

- [3ML](https://threeml.readthedocs.io)
- [hawc_hal](https://github.com/threeML/hawc_hal)
- [Ollama](https://ollama.com) with `llama3.1:8b` (for OPRO scripts)

## Usage

**Standard fit:**
```bash
python scripts/crab.py
```

**OPRO fit:**
```bash
# Requires Ollama running on localhost:11435
python scripts/crab_opro_change_start_v6.py
```

Data files must be present in `data/`. Update the `maptree` and `response` paths in the scripts if needed.
