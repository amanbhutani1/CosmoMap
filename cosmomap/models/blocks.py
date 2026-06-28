"""Building blocks for the 3D U-Net: the conditioning MLP and the FiLM residual
block (GroupNorm + 3D convolutions + FiLM modulation + spatial dropout)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from cosmomap.models.film import FiLM3D


def conditioning_network(cond_input: tf.Tensor, hidden: int) -> tf.Tensor:
    """Map the raw conditioning vector to a richer embedding for FiLM."""
    h = layers.Dense(hidden, activation="relu", name="cond_dense_1")(cond_input)
    h = layers.Dense(hidden, activation="relu", name="cond_dense_2")(h)
    return h


def res_block_3d(
    x: tf.Tensor,
    cond: tf.Tensor,
    filters: int,
    groups: int,
    dropout: float,
    name: str,
    convs: int = 2,
) -> tf.Tensor:
    """Residual block: ``convs`` x (Conv-GN-[ReLU]-FiLM) + spatial dropout + skip.

    GroupNorm is used instead of BatchNorm so behaviour is independent of batch
    size (important for the small batches used with large 3D volumes).
    """
    g = min(groups, filters)
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv3D(filters, 1, padding="same", name=f"{name}_proj")(x)

    h = x
    for j in range(convs):
        h = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv{j + 1}")(h)
        h = layers.GroupNormalization(groups=g, name=f"{name}_gn{j + 1}")(h)
        if j < convs - 1:  # last activation comes after the residual add
            h = layers.Activation("relu", name=f"{name}_relu{j + 1}")(h)
        h = FiLM3D(name=f"{name}_film{j + 1}")([h, cond])

    if dropout > 0:
        h = layers.SpatialDropout3D(dropout, name=f"{name}_drop")(h)

    h = layers.Add(name=f"{name}_add")([shortcut, h])
    return layers.Activation("relu", name=f"{name}_out")(h)
