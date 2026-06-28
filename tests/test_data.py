import numpy as np

from cosmomap.data import make_dataset, make_synthetic_camels, split_indices


def test_synthetic_shapes(tiny_config):
    data = make_synthetic_camels(tiny_config, n_sims=3, seed=0)
    assert data.n_samples == 3 * len(tiny_config.data.redshifts)
    r = tiny_config.data.resolution
    for f in tiny_config.data.fields_in + tiny_config.data.fields_out:
        assert data.fields[f].shape == (data.n_samples, r, r, r)
    assert data.cond.shape == (data.n_samples, tiny_config.n_cond)
    assert np.isfinite(data.fields["ne"]).all()
    assert (data.fields["ne"] > 0).all()  # electron density is strictly positive


def test_dataset_batch_shapes(tiny_config):
    data = make_synthetic_camels(tiny_config, n_sims=2, seed=0)
    ds = make_dataset(data, np.arange(data.n_samples), tiny_config, training=False)
    inputs, targets = next(iter(ds))
    p = tiny_config.data.patch_size
    bs = tiny_config.train.batch_size
    assert tuple(inputs["image"].shape) == (bs, p, p, p, tiny_config.n_in)
    assert tuple(inputs["cosmo"].shape) == (bs, tiny_config.n_cond)
    assert tuple(targets.shape) == (bs, p, p, p, tiny_config.n_out)


def test_split_indices_partition():
    train, val, test = split_indices(100, [0.8, 0.1, 0.1], seed=1)
    allidx = np.concatenate([train, val, test])
    assert len(allidx) == 100
    assert len(set(allidx.tolist())) == 100  # disjoint, covering
