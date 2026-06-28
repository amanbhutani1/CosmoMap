import numpy as np

from cosmomap.inference import build_feature_matrix, extract_features, recover_parameters


def _field(rng, shape=(8, 8, 8)):
    return np.abs(rng.standard_normal(shape)) + 1.0  # strictly positive


def test_feature_vector_is_32d():
    rng = np.random.default_rng(0)
    feats = extract_features(_field(rng), _field(rng))
    assert feats.shape == (32,)
    assert np.all(np.isfinite(feats))


def test_recover_parameters_runs():
    rng = np.random.default_rng(0)
    samples = [{"T": _field(rng), "ne": _field(rng)} for _ in range(40)]
    features = build_feature_matrix(samples)
    assert features.shape == (40, 32)
    targets = np.stack([features[:, 0] * 2.0, features[:, 4]], axis=1)
    _, scores = recover_parameters(features, targets, ["Omega_m", "sigma_8"], n_estimators=15)
    assert set(scores) == {"Omega_m", "sigma_8"}
    assert all(np.isfinite(v) for v in scores.values())
