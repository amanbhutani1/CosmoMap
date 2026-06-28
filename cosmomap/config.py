"""Typed, serialisable configuration for the CosmoMap pipeline.

All hyperparameters live here so that experiments are fully described by a single
YAML file. Defaults reproduce the configuration reported in the dissertation
(patch size 80, 3-level encoder, FiLM conditioning on six CAMELS parameters,
heteroscedastic outputs, loss weights from the hyperparameter search).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    """Dataset, field and preprocessing settings."""

    data_root: str = "data/camels"
    #: Redshift snapshots used for (multi-z) training.
    redshifts: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    #: Input channels: cold-dark-matter mass density and velocity magnitude.
    fields_in: list[str] = field(default_factory=lambda: ["Mcdm", "Vcdm"])
    #: Target channels: gas temperature and electron number density.
    fields_out: list[str] = field(default_factory=lambda: ["T", "ne"])
    #: CAMELS parameters used for FiLM conditioning (2 cosmological + 4 feedback).
    cosmo_params: list[str] = field(
        default_factory=lambda: ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    )
    resolution: int = 128  # native voxels per simulation side (25 h^-1 Mpc box)
    box_size_mpc_h: float = 25.0
    patch_size: int = 80  # cubic sub-volume fed to the network
    include_redshift_in_cond: bool = False  # thesis conditions on the 6 params only (Eq 7)
    log_epsilon: float = 1e-8  # floor for the log10 transform of multiplicative fields
    train_val_test_split: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])


@dataclass
class ModelConfig:
    """3D FiLM-conditioned U-Net architecture settings."""

    base_filters: int = 64  # channels at the first encoder level (64 -> 128 -> 256)
    depth: int = 3  # number of encoder/decoder levels
    # Light encoder/decoder blocks + a heavier bottleneck (~11M parameters total,
    # with the bottleneck as the heaviest component).
    convs_per_block: int = 1  # 3x3x3 convolutions per encoder/decoder residual block
    bottleneck_convs: int | None = 2  # convs in the bottleneck block
    groups: int = 8  # GroupNorm groups
    cond_hidden: int = 192  # width of the conditioning MLP (FiLM generator)
    dropout: float = 0.1  # spatial dropout; also used for MC-dropout epistemic UQ
    heteroscedastic: bool = True  # predict per-voxel log-variance alongside the mean


@dataclass
class LossConfig:
    """Composite-loss weights (optimised in the dissertation's HPO search)."""

    w_nll: float = 0.408  # heteroscedastic negative log-likelihood
    w_mse: float = 0.150
    w_mae: float = 0.134
    w_huber: float = 0.207
    huber_delta: float = 1.0
    lambda_positivity: float = 0.1  # penalise negative electron-density predictions
    lambda_variance: float = 0.01  # regularise predicted variance (prevents divergence)


@dataclass
class TrainConfig:
    """Optimisation and runtime settings."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"  # one of {"cosine", "constant"}
    grad_clip_norm: float = 1.0  # gradient clipping (tau = 1.0 in the thesis)
    early_stop_patience: int = 10
    seed: int = 42
    mixed_precision: bool = True
    multi_gpu: bool = False  # wrap training in tf.distribute.MirroredStrategy
    steps_per_epoch: int | None = None  # None -> derived from the dataset size
    output_dir: str = "outputs"


@dataclass
class Config:
    """Top-level configuration aggregating every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- Derived shapes -----------------------------------------------------
    @property
    def n_cond(self) -> int:
        """Length of the FiLM conditioning vector."""
        return len(self.data.cosmo_params) + int(self.data.include_redshift_in_cond)

    @property
    def n_in(self) -> int:
        """Number of input channels."""
        return len(self.data.fields_in)

    @property
    def n_out(self) -> int:
        """Number of predicted physical fields (mean channels)."""
        return len(self.data.fields_out)

    @property
    def out_channels(self) -> int:
        """Network output channels: mean (+ log-variance if heteroscedastic)."""
        return self.n_out * (2 if self.model.heteroscedastic else 1)

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        p = self.data.patch_size
        return (p, p, p, self.n_in)

    # -- (De)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with Path(path).open() as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


def _build(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate (possibly nested) dataclasses from a plain dict.

    Unknown keys are ignored so that configs remain forward-compatible.
    """
    # ``get_type_hints`` resolves the string annotations produced by
    # ``from __future__ import annotations`` back to real classes.
    hints = get_type_hints(dc_type)
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)
