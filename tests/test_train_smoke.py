"""End-to-end smoke test: one training epoch on synthetic data must run and
produce a saved model."""

from cosmomap.train import train


def test_train_one_epoch(tiny_config, tmp_path):
    tiny_config.train.output_dir = str(tmp_path / "out")
    tiny_config.train.epochs = 1
    _, history = train(tiny_config, synthetic=True, n_sims=8)
    assert "loss" in history.history
    assert (tmp_path / "out" / "final_model.keras").exists()
    assert (tmp_path / "out" / "config.yaml").exists()
