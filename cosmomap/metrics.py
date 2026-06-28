"""Evaluation metrics.

Two flavours:

* :func:`keras_metrics` returns TensorFlow metric callables (sliced onto the
  predicted *mean* channels) for use in ``model.compile`` / ``model.fit``;
* the NumPy functions (:func:`r2_score`, :func:`rmse`, :func:`pearson_r`,
  :func:`psnr`, :func:`ssim3d`, :func:`power_spectrum_3d`) operate on plain
  arrays and are used by :mod:`cosmomap.evaluate`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter

from cosmomap.config import Config


# --------------------------------------------------------------------------- #
# Keras (graph) metrics — evaluated on the predicted mean channels.
# --------------------------------------------------------------------------- #
def keras_metrics(config: Config) -> list:
    """Return compile-ready metrics that slice the mean out of the prediction."""
    n = config.n_out
    het = config.model.heteroscedastic

    def _mean(y_pred: tf.Tensor) -> tf.Tensor:
        return y_pred[..., :n] if het else y_pred

    def rmse(y_true, y_pred):
        m = _mean(tf.cast(y_pred, tf.float32))
        return tf.sqrt(tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - m)))

    def mae(y_true, y_pred):
        m = _mean(tf.cast(y_pred, tf.float32))
        return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - m))

    def r2(y_true, y_pred):
        yt = tf.cast(y_true, tf.float32)
        m = _mean(tf.cast(y_pred, tf.float32))
        ss_res = tf.reduce_sum(tf.square(yt - m))
        ss_tot = tf.reduce_sum(tf.square(yt - tf.reduce_mean(yt)))
        return 1.0 - ss_res / (ss_tot + 1e-8)

    def psnr(y_true, y_pred):
        yt = tf.cast(y_true, tf.float32)
        m = _mean(tf.cast(y_pred, tf.float32))
        mse = tf.reduce_mean(tf.square(yt - m))
        data_range = tf.reduce_max(yt) - tf.reduce_min(yt) + 1e-8
        return 20.0 * tf.math.log(data_range / tf.sqrt(mse + 1e-12)) / tf.math.log(10.0)

    for fn, fname in [(rmse, "rmse"), (mae, "mae"), (r2, "r2"), (psnr, "psnr")]:
        fn.__name__ = fname
    return [rmse, mae, r2, psnr]


# --------------------------------------------------------------------------- #
# NumPy metrics for offline evaluation.
# --------------------------------------------------------------------------- #
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-12
    return float(np.sum(a * b) / denom)


def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float | None = None) -> float:
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = y_true.max() - y_true.min() + 1e-12
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def ssim3d(
    y_true: np.ndarray, y_pred: np.ndarray, window: int = 7, data_range: float | None = None
) -> float:
    """Mean structural similarity over a 3D volume (uniform-window SSIM)."""
    a, b = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    if data_range is None:
        data_range = a.max() - a.min() + 1e-12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = uniform_filter(a, window)
    mu_b = uniform_filter(b, window)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    var_a = uniform_filter(a * a, window) - mu_a2
    var_b = uniform_filter(b * b, window) - mu_b2
    cov_ab = uniform_filter(a * b, window) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map.mean())


def power_spectrum_3d(field: np.ndarray, box_size: float = 1.0, n_bins: int = 32):
    """Isotropic 3D power spectrum ``P(k)`` via spherical averaging of |FFT|^2.

    Returns ``(k, Pk)`` with ``k`` in units of ``2*pi / box_size``.
    """
    field = np.asarray(field, np.float64)
    field = field - field.mean()
    n = field.shape[0]
    fk = np.fft.fftn(field)
    power = np.abs(fk) ** 2 / field.size

    kfreq = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    power = power.ravel()

    k_edges = np.linspace(0, kmag.max() + 1e-9, n_bins + 1)
    which = np.digitize(kmag, k_edges) - 1
    pk = np.array([power[which == i].mean() if np.any(which == i) else 0.0 for i in range(n_bins)])
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1]) * (2 * np.pi / box_size)
    return k_centers, pk
