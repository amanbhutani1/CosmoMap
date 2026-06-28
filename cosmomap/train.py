"""Training entry point.

Usage
-----
Synthetic demo (no data needed)::

    cosmomap-train --config configs/synthetic.yaml --synthetic

Real CAMELS data::

    cosmomap-train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from cosmomap.config import Config
from cosmomap.data import make_dataset, make_synthetic_camels, split_simulations
from cosmomap.data.camels import load_camels
from cosmomap.data.container import CamelsData
from cosmomap.losses import make_loss
from cosmomap.metrics import keras_metrics
from cosmomap.models import build_model


def set_global_seed(seed: int) -> None:
    # Seeds Python, NumPy and TensorFlow RNGs in one call.
    keras.utils.set_random_seed(seed)


def build_optimizer(config: Config, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    tc = config.train
    if tc.lr_schedule == "cosine":
        schedule = keras.optimizers.schedules.CosineDecay(
            tc.learning_rate, decay_steps=max(1, steps_per_epoch * tc.epochs)
        )
        learning_rate: float | keras.optimizers.schedules.LearningRateSchedule = schedule
    else:
        learning_rate = tc.learning_rate
    # Under a mixed_float16 policy, Keras 3 applies loss scaling automatically.
    return keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=tc.grad_clip_norm)


def load_data(config: Config, synthetic: bool, n_sims: int) -> CamelsData:
    if synthetic:
        return make_synthetic_camels(config, n_sims=n_sims, seed=config.train.seed)
    return load_camels(config)


def train(config: Config, synthetic: bool = False, n_sims: int = 16):
    """Train the model and write artefacts to ``config.train.output_dir``."""
    set_global_seed(config.train.seed)
    if config.train.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    data = load_data(config, synthetic, n_sims)
    train_idx, val_idx, _ = split_simulations(
        data.sim_ids, config.data.train_val_test_split, config.train.seed
    )
    train_ds = make_dataset(data, train_idx, config, training=True)
    val_ds = make_dataset(data, val_idx, config, training=False)
    steps = config.train.steps_per_epoch or max(1, len(train_idx) // config.train.batch_size)

    def _compile() -> keras.Model:
        model = build_model(config)
        model.compile(
            optimizer=build_optimizer(config, steps),
            loss=make_loss(config),
            metrics=keras_metrics(config),
        )
        return model

    if config.train.multi_gpu:
        with tf.distribute.MirroredStrategy().scope():
            model = _compile()
    else:
        model = _compile()

    out_dir = Path(config.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")
    model.summary(print_fn=lambda line: print(line))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.train.early_stop_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.train.epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        shuffle=False,  # batching is handled by the (already-shuffled) tf.data pipeline
        verbose=2,
    )

    model.save(out_dir / "final_model.keras")
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2)
    )
    print(f"\nSaved model and logs to {out_dir.resolve()}")
    return model, history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the CosmoMap 3D FiLM U-Net.")
    parser.add_argument("--config", default="configs/synthetic.yaml", help="path to a YAML config")
    parser.add_argument(
        "--synthetic", action="store_true", help="use generated data (no download required)"
    )
    parser.add_argument("--n-sims", type=int, default=16, help="number of synthetic simulations")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir

    train(config, synthetic=args.synthetic, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
