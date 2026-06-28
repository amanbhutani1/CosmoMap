from cosmomap.config import Config


def test_defaults_match_thesis():
    c = Config()
    assert c.data.patch_size == 80
    assert c.data.cosmo_params == ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    assert c.n_cond == 6  # 6 CAMELS parameters (no redshift, per Eq 7)
    assert c.n_in == 2 and c.n_out == 2
    assert c.out_channels == 4  # heteroscedastic: 2 means + 2 log-variances
    assert c.input_shape == (80, 80, 80, 2)


def test_yaml_roundtrip(tmp_path):
    c = Config()
    path = tmp_path / "config.yaml"
    c.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.loss.w_nll == c.loss.w_nll
    assert loaded.data.redshifts == c.data.redshifts
    assert loaded.model.heteroscedastic is True
    assert loaded.n_cond == c.n_cond


def test_unknown_keys_ignored():
    c = Config.from_dict({"data": {"patch_size": 64, "nonsense": 1}, "extra": True})
    assert c.data.patch_size == 64
