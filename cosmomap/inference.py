"""Cosmological-parameter recovery (§3.4.3).

Extracts a **32-dimensional** feature vector from predicted (or true) fields and
trains a Random-Forest regressor to recover the CAMELS parameters (primarily
Ω_m and σ₈).

Feature vector (32-d):
  * statistical moments (mean, std, skewness, kurtosis) of T, n_e and the X-ray
    emissivity ε_X .................................................... 12
  * power-spectrum amplitudes at k = {1, 2, 5, 10, 20} h/Mpc for T, n_e, ε_X . 15
  * morphology: cluster count, mean separation, filling factor, total
    luminosity, largest-cluster volume ................................. 5
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from cosmomap.metrics import power_spectrum_3d, r2_score
from cosmomap.validation import detect_clusters
from cosmomap.xray.emissivity import compute_emissivity

#: Power-spectrum wavenumbers sampled as features (h/Mpc).
K_VALUES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)


def _moments(field: np.ndarray) -> list[float]:
    f = np.asarray(field, dtype=np.float64).ravel()
    mu = float(f.mean())
    sd = float(f.std()) + 1e-12
    z = (f - mu) / sd
    return [mu, sd, float(np.mean(z**3)), float(np.mean(z**4))]  # mean, std, skew, kurtosis


def _power_at_k(field: np.ndarray, box_size: float) -> list[float]:
    k, pk = power_spectrum_3d(field, box_size=box_size)
    good = k > 0
    return [float(np.interp(kv, k[good], pk[good])) for kv in K_VALUES]


def _morphology(emissivity: np.ndarray, box_size: float) -> list[float]:
    _, n, props = detect_clusters(emissivity, percentile=90.0)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    volumes = np.array([p["volume"] for p in props], dtype=np.float64)
    lums = np.array([p["luminosity"] for p in props], dtype=np.float64)
    voxel = box_size / emissivity.shape[0]
    if n > 1:
        mean_sep = float(pdist(np.array([p["centroid"] for p in props])).mean()) * voxel
    else:
        mean_sep = 0.0
    filling = float(volumes.sum() / emissivity.size)
    return [float(n), mean_sep, filling, float(lums.sum()), float(volumes.max())]


def extract_features(
    temperature: np.ndarray, ne: np.ndarray, box_size: float = 25.0
) -> np.ndarray:
    """Build the 32-d feature vector from a predicted (T, n_e) pair."""
    eps = compute_emissivity(ne, temperature)
    feats: list[float] = []
    for field in (temperature, ne, eps):
        feats += _moments(field)  # 12
    for field in (temperature, ne, eps):
        feats += _power_at_k(field, box_size)  # 15
    feats += _morphology(eps, box_size)  # 5
    return np.asarray(feats, dtype=np.float64)


def build_feature_matrix(
    samples: Sequence[dict[str, np.ndarray]], box_size: float = 25.0
) -> np.ndarray:
    """Stack 32-d feature vectors for samples; each sample needs ``T`` and ``ne``."""
    return np.stack([extract_features(s["T"], s["ne"], box_size) for s in samples])


def recover_parameters(
    features: np.ndarray,
    targets: np.ndarray,
    param_names: Sequence[str],
    *,
    test_size: float = 0.2,
    n_estimators: int = 200,
    seed: int = 42,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Fit a Random Forest (trained on 80% of simulations) and report test R²."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=seed
    )
    forest = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    forest.fit(x_train, y_train)
    predictions = forest.predict(x_test)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        y_test = np.asarray(y_test).reshape(-1, 1)
    scores = {
        name: r2_score(y_test[:, i], predictions[:, i]) for i, name in enumerate(param_names)
    }
    return forest, scores
