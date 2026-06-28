import numpy as np
import tensorflow as tf

from cosmomap.data.normalization import denormalize_field, normalize_field
from cosmomap.losses import make_loss
from cosmomap.metrics import power_spectrum_3d, r2_score, rmse, ssim3d


def test_loss_finite_and_nonnegative(tiny_config):
    loss = make_loss(tiny_config)
    p, n = tiny_config.data.patch_size, tiny_config.n_out
    y_true = tf.random.normal((2, p, p, p, n))
    y_pred = tf.random.normal((2, p, p, p, tiny_config.out_channels))
    value = float(loss(y_true, y_pred).numpy())
    assert np.isfinite(value) and value >= 0.0


def test_loss_prefers_correct_mean(tiny_config):
    loss = make_loss(tiny_config)
    p, n = tiny_config.data.patch_size, tiny_config.n_out
    y_true = tf.random.normal((2, p, p, p, n))
    zeros_logvar = tf.zeros_like(y_true)
    good = tf.concat([y_true, zeros_logvar], axis=-1)
    bad = tf.concat([y_true + 1.0, zeros_logvar], axis=-1)
    assert float(loss(y_true, good).numpy()) < float(loss(y_true, bad).numpy())


def test_normalization_roundtrip():
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((8, 8, 8)).astype(np.float32)) + 0.1
    normed, stats = normalize_field(x, "ne")  # log-transformed field
    recovered = denormalize_field(normed, stats)
    assert np.allclose(x, recovered, rtol=1e-2, atol=1e-3)


def test_metrics_on_perfect_prediction():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4, 8, 8)).astype(np.float32)
    assert r2_score(y, y) > 0.999
    assert rmse(y, y) < 1e-6
    assert ssim3d(y, y) > 0.999


def test_power_spectrum_runs():
    rng = np.random.default_rng(0)
    k, pk = power_spectrum_3d(rng.standard_normal((16, 16, 16)))
    assert k.shape == pk.shape
    assert np.all(np.isfinite(pk))
