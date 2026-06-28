import numpy as np

from cosmomap.xray import (
    EMISSIVITY_CONST,
    LightconeConfig,
    build_lightcone,
    compute_emissivity,
    save_fits,
)


def test_emissivity_formula():
    # eps = A * ne^2 * sqrt(T) = 1.42e-27 * 4 * 2
    e = compute_emissivity(np.array([2.0]), np.array([4.0]))
    assert np.isclose(e[0], EMISSIVITY_CONST * 4.0 * 2.0)


def test_lightcone_and_fits(tmp_path):
    rng = np.random.default_rng(0)
    mk = lambda: np.abs(rng.standard_normal((8, 8, 8))) + 0.1  # noqa: E731
    ne = {0.0: mk(), 1.0: mk()}
    temperature = {0.0: mk() * 1e6, 1.0: mk() * 1e6}
    cfg = LightconeConfig(redshifts=[0.0, 1.0], npix=16, box_size_mpc_h=25.0)
    sb = build_lightcone(ne, temperature, cfg)
    assert sb.shape == (16, 16)
    assert np.all(np.isfinite(sb))

    path = tmp_path / "lightcone.fits"
    save_fits(sb, path, cfg)
    assert path.exists()
