"""3D FiLM-conditioned U-Net with heteroscedastic output heads.

The network maps a multi-channel dark-matter sub-box plus a conditioning vector
to baryonic fields. With ``heteroscedastic=True`` the output has ``2 * n_out``
channels: the first ``n_out`` are the predicted means and the remaining ``n_out``
are predicted log-variances, enabling aleatoric uncertainty.

Architecture (§3.3.2): a 3-level encoder/decoder with
channels ``64 -> 128 -> 256`` and the bottleneck held at the deepest level (256),
GroupNorm (8 groups), residual blocks with FiLM conditioning, dense skip
connections, and no attention. The patch size must be divisible by
``2 ** depth`` so the encoder/decoder spatial dimensions line up (e.g. 80 with
depth 3 -> 10 at the bottleneck).
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from cosmomap.config import Config
from cosmomap.models.blocks import conditioning_network, res_block_3d


def build_model(config: Config) -> keras.Model:
    """Construct the 3D FiLM U-Net described by ``config``."""
    p = config.data.patch_size
    depth = config.model.depth
    if p % (2**depth) != 0:
        raise ValueError(
            f"patch_size ({p}) must be divisible by 2**depth ({2**depth}) for the U-Net."
        )

    mc = config.model
    # Channel counts per encoder level: 64 -> 128 -> 256 (deepest == bottleneck).
    enc_filters = [mc.base_filters * (2**i) for i in range(depth)]

    image = keras.Input(shape=(p, p, p, config.n_in), name="image")
    cosmo = keras.Input(shape=(config.n_cond,), name="cosmo")
    cond = conditioning_network(cosmo, mc.cond_hidden)

    # Stem
    x = layers.Conv3D(mc.base_filters, 3, padding="same", name="stem_conv")(image)
    x = layers.GroupNormalization(groups=min(mc.groups, mc.base_filters), name="stem_gn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # Encoder
    skips: list = []
    for i, f in enumerate(enc_filters):
        x = res_block_3d(x, cond, f, mc.groups, mc.dropout, name=f"enc{i}", convs=mc.convs_per_block)
        skips.append(x)
        x = layers.MaxPooling3D(2, name=f"enc{i}_pool")(x)

    # Bottleneck held at the deepest channel count (no extra doubling to 512).
    bottleneck_convs = mc.bottleneck_convs if mc.bottleneck_convs is not None else mc.convs_per_block
    x = res_block_3d(
        x, cond, enc_filters[-1], mc.groups, mc.dropout, name="bottleneck", convs=bottleneck_convs
    )

    # Decoder (mirror of the encoder)
    for i, f in enumerate(reversed(enc_filters)):
        x = layers.Conv3DTranspose(f, 2, strides=2, padding="same", name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_concat")([x, skips.pop()])
        x = res_block_3d(x, cond, f, mc.groups, mc.dropout, name=f"dec{i}", convs=mc.convs_per_block)

    # Output head(s). Kept in float32 for numerical stability under mixed precision.
    output = layers.Conv3D(
        config.out_channels, 1, padding="same", name="prediction", dtype="float32"
    )(x)

    return keras.Model(inputs={"image": image, "cosmo": cosmo}, outputs=output, name="cosmomap_unet")
