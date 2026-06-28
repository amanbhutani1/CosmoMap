import numpy as np

from cosmomap.validation import (
    beta_model,
    detect_clusters,
    fit_beta_model,
    mahalanobis_distance,
    scaling_relation,
)


def test_scaling_relation_recovers_exponent():
    rng = np.random.default_rng(0)
    dm = np.abs(rng.standard_normal((16, 16, 16))) + 0.1
    field = dm**1.5 * 10 ** (0.03 * rng.standard_normal((16, 16, 16)))  # field ∝ dm^1.5
    res = scaling_relation(dm, field, n_samples=2000)
    assert abs(res["exponent"] - 1.5) < 0.15
    assert res["r2"] > 0.8


def test_mahalanobis_distance_shape_and_sign():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((100, 6))
    d = mahalanobis_distance(rng.standard_normal((5, 6)), train)
    assert d.shape == (5,)
    assert np.all(d >= 0)


def test_detect_clusters_finds_blobs():
    field = np.zeros((16, 16, 16))
    field[4:7, 4:7, 4:7] = 10.0
    field[10:12, 10:12, 10:12] = 8.0
    _, n, props = detect_clusters(field, percentile=95)
    assert n >= 1
    assert all("volume" in p and "luminosity" in p for p in props)


def test_beta_model_roundtrip():
    r = np.linspace(0.1, 10, 50)
    sb = beta_model(r, 100.0, 2.0, 0.6)
    fit = fit_beta_model(r, sb)
    assert abs(fit["beta"] - 0.6) < 0.1
    assert abs(fit["r_c"] - 2.0) < 0.5
