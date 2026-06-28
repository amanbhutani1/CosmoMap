# Method

A write-up of the CosmoMap method, mapping each component to its governing
equation and to the code that implements it.

## Problem

Hydrodynamic simulations are expensive. We learn a fast surrogate that maps the
(cheap) dark-matter fields to the (expensive) baryonic fields, conditioned on the
simulation's cosmological/astrophysical parameters, then push the predictions
through an X-ray forward model.

| Symbol | Meaning | Role |
|---|---|---|
| `Mcdm` | dark-matter mass density | input channel |
| `Vcdm` | dark-matter velocity magnitude | input channel |
| `T` | gas temperature | target |
| `ne` | electron number density | target |
| `Omega_m, sigma_8` | cosmological parameters | FiLM conditioning |
| `A_SN1, A_AGN1, A_SN2, A_AGN2` | supernova/AGN feedback | FiLM conditioning |

## Pipeline → code

| Stage | Description | Module |
|---|---|---|
| Normalisation | `log10` of multiplicative fields, then per-patch z-score (invertible) | `cosmomap/data/normalization.py` |
| Patch sampling | random crop (train) / centred crop (eval); **simulation-level split** | `cosmomap/data/dataset.py` |
| Conditioning | MLP on the **6 CAMELS params** (Eq 7) → FiLM embedding | `cosmomap/models/blocks.py` |
| FiLM | per-channel affine modulation `y = (1+γ)·x + β` | `cosmomap/models/film.py` |
| Backbone | 3D residual U-Net, **64→128→256** (bottleneck 256), GroupNorm-8 + FiLM, **~11.1M params** | `cosmomap/models/unet3d.py` |
| Heteroscedastic head | predicts mean **and** `log σ²` per field (4-ch output) | `cosmomap/models/unet3d.py` |
| Loss | `w_nll·NLL + w_mse·MSE + w_mae·MAE + w_huber·Huber + physics` | `cosmomap/losses.py` |
| Uncertainty | aleatoric (`log σ²`) + epistemic (20× MC-dropout) | `cosmomap/uncertainty.py` |
| Evaluation (§3.4.4–3.4.5) | overall + density-stratified R²/bias + calibration | `cosmomap/evaluate.py` |
| Scaling relations (§3.4.1) | `ne ∝ ρ^α`, `T ∝ ρ^β` via log-log regression | `cosmomap/validation.py` |
| Cluster detection (§3.4.2) | 26-connectivity components + β-model fit | `cosmomap/validation.py` |
| OOD robustness (§3.4.6) | Mahalanobis distance from training params | `cosmomap/validation.py` |
| X-ray emissivity | `ε_X = 1.42e-27 · ne² · √T` | `cosmomap/xray/emissivity.py` |
| Lightcone (§3.5.2) | D_A angular mapping, trilinear z-interp, `(1+z)⁻⁴` dimming, PSF, background | `cosmomap/xray/lightcone.py` |
| Parameter recovery (§3.4.3) | Random Forest on a **32-d** feature vector (moments + P(k) + morphology) | `cosmomap/inference.py` |

## Architecture

A 3-level `64→128→256` FiLM-conditioned U-Net (~11M parameters): the bottleneck is
held at 256, GroupNorm (8 groups), residual blocks with FiLM modulation, dense skip
connections, no attention, and a 4-channel heteroscedastic output. Light
encoder/decoder blocks with a heavier bottleneck (`convs_per_block=1`,
`bottleneck_convs=2`) concentrate capacity where the field interactions are densest.

## Key equations

- **Normalisation:** `X' = (log10(max(X, ε)) − μ_patch) / σ_patch`
- **Composite loss:** `L = 0.408·NLL + 0.150·MSE + 0.134·MAE + 0.207·Huber + λ_pos·E[max(0,−µ_ne)] + λ_var·E[σ²]`
- **Gaussian NLL:** `NLL = ½·E[(y−µ)²/σ² + log σ²]`
- **X-ray emissivity (bremsstrahlung):** `ε_X = 1.42×10⁻²⁷ · ne² · √T`

## Results

| Quantity | Value |
|---|---|
| Overall R² | 0.840 |
| Gas density (ne) R² | 0.947 |
| Temperature R² | 0.733 |
| Temperature R²: void → cluster core | 0.548 → 0.796 |
| Density R²: void → cluster core | 0.627 → 0.835 |
| X-ray emissivity correlation | r = 0.963 ± 0.007 |
| 95% predictive-interval coverage | 94.2% |
| Parameter recovery (Ω_m, σ₈) R² | ≈ 0.64 |
