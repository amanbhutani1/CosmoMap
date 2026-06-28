"""Uncertainty quantification.

The model captures two kinds of uncertainty:

* **aleatoric** (data noise) — the predicted ``log_var`` head, ``sigma^2 = exp(log_var)``;
* **epistemic** (model uncertainty) — estimated with Monte-Carlo dropout: run the
  network ``n_passes`` times with dropout active and take the variance of the
  predicted means.

The total predictive variance is their sum.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

_LOG_VAR_CLIP = 10.0


def mc_dropout_predict(
    model: tf.keras.Model,
    inputs: dict,
    n_passes: int = 20,
    n_out: int = 2,
    heteroscedastic: bool = True,
) -> dict[str, np.ndarray]:
    """Monte-Carlo dropout prediction with aleatoric/epistemic decomposition.

    Returns a dict with keys ``mean``, ``epistemic``, ``aleatoric`` and ``total``
    (each shaped like a single prediction's mean channels).
    """
    means, variances = [], []
    for _ in range(n_passes):
        out = tf.cast(model(inputs, training=True), tf.float32).numpy()
        means.append(out[..., :n_out])
        if heteroscedastic:
            variances.append(np.exp(np.clip(out[..., n_out:], -_LOG_VAR_CLIP, _LOG_VAR_CLIP)))

    means_arr = np.stack(means, axis=0)
    mean = means_arr.mean(axis=0)
    epistemic = means_arr.var(axis=0)
    aleatoric = (
        np.stack(variances, axis=0).mean(axis=0) if heteroscedastic else np.zeros_like(mean)
    )
    return {
        "mean": mean,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": epistemic + aleatoric,
    }


def calibration_coverage(
    y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray, n_sigma: float = 1.96
) -> float:
    """Fraction of true values inside ``mean +/- n_sigma * sqrt(variance)``.

    For a well-calibrated Gaussian predictive distribution this should be ~0.95
    at ``n_sigma = 1.96``.
    """
    sigma = np.sqrt(np.asarray(variance) + 1e-12)
    lo, hi = mean - n_sigma * sigma, mean + n_sigma * sigma
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside))
