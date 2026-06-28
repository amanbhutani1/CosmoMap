"""Scientific validation analyses from the dissertation's §3.4 framework:

* scaling relations (§3.4.1): n_e ∝ ρ_DM^α and T ∝ ρ_DM^β via log-log regression,
* cluster detection (§3.4.2): 3D connected components on X-ray emissivity + a
  β-model surface-brightness fit,
* parameter-space robustness (§3.4.6): Mahalanobis distance from the training set.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize


def scaling_relation(
    dm_density: np.ndarray, field: np.ndarray, n_samples: int = 10_000, seed: int = 42
) -> dict[str, float]:
    """Fit a power law ``field ∝ rho_DM^exponent`` by log-log regression on up to
    ``n_samples`` randomly sampled voxels (§3.4.1). Returns exponent, intercept, R².
    """
    rng = np.random.default_rng(seed)
    dm = np.asarray(dm_density, dtype=np.float64).ravel()
    fl = np.asarray(field, dtype=np.float64).ravel()
    mask = (dm > 0) & (fl > 0)
    dm, fl = dm[mask], fl[mask]
    if dm.size > n_samples:
        idx = rng.choice(dm.size, n_samples, replace=False)
        dm, fl = dm[idx], fl[idx]
    x, y = np.log10(dm), np.log10(fl)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / (ss_tot + 1e-12)),
    }


def mahalanobis_distance(p_test: np.ndarray, p_train: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of test parameter vectors from the training
    distribution (§3.4.6, Eq 16), using the (pseudo-inverse) training covariance.
    """
    p_train = np.asarray(p_train, dtype=np.float64)
    mu = p_train.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(p_train, rowvar=False))
    p_test = np.atleast_2d(np.asarray(p_test, dtype=np.float64))
    delta = p_test - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, cov_inv, delta), 0.0))


def detect_clusters(
    emissivity: np.ndarray, percentile: float = 90.0
) -> tuple[np.ndarray, int, list[dict]]:
    """3D connected-component cluster detection (§3.4.2).

    Thresholds the emissivity field at ``percentile`` and labels components with
    full 26-connectivity. Returns ``(labels, n_clusters, properties)``.
    """
    emis = np.asarray(emissivity, dtype=np.float64)
    mask = emis > np.percentile(emis, percentile)
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labels, n = ndimage.label(mask, structure=structure)
    props = []
    for i in range(1, n + 1):
        sel = labels == i
        props.append(
            {
                "volume": int(sel.sum()),
                "centroid": [float(c) for c in ndimage.center_of_mass(sel)],
                "luminosity": float(emis[sel].sum()),
            }
        )
    return labels, n, props


def beta_model(r: np.ndarray, s0: float, r_c: float, beta: float) -> np.ndarray:
    """Canonical X-ray surface-brightness β-model (Eq 15):
    ``S(r) = S0 * [1 + (r/r_c)^2]^(-3β + 1/2)``."""
    r = np.asarray(r, dtype=np.float64)
    return s0 * (1.0 + (r / r_c) ** 2) ** (-3.0 * beta + 0.5)


def fit_beta_model(radii: np.ndarray, surface_brightness: np.ndarray) -> dict[str, float]:
    """Fit the β-model to a radial surface-brightness profile; returns S0, r_c, β."""
    radii = np.asarray(radii, dtype=np.float64)
    sb = np.asarray(surface_brightness, dtype=np.float64)
    p0 = [float(np.max(sb)), max(float(np.mean(radii)), 1.0), 0.6]
    try:
        popt, _ = optimize.curve_fit(beta_model, radii, sb, p0=p0, maxfev=10_000)
        s0, r_c, beta = (float(v) for v in popt)
    except (RuntimeError, ValueError):
        s0, r_c, beta = p0
    return {"S0": s0, "r_c": r_c, "beta": beta}
