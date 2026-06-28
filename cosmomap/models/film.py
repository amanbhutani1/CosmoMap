"""Feature-wise Linear Modulation (FiLM) for 3D feature maps.

FiLM conditions the network on the CAMELS parameter vector by predicting a
per-channel affine transform of intermediate feature maps (Perez et al. 2018).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="cosmomap")
class FiLM3D(layers.Layer):
    r"""Per-channel affine modulation of a rank-5 ``(B, D, H, W, C)`` tensor.

    Given a feature map ``x`` and a conditioning vector ``cond``::

        y = (1 + gamma(cond)) * x + beta(cond)

    where ``gamma`` and ``beta`` are dense maps from the conditioning vector to
    ``C`` channels. They are initialised small so the block starts close to the
    identity (stable early training) while still conditioning on ``cond``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        feature_shape, cond_shape = input_shape
        channels = int(feature_shape[-1])
        # Small (not zero) init: starts near identity but still conditions on cond.
        init = tf.keras.initializers.RandomNormal(stddev=0.1)
        # Pin sublayers to this layer's dtype policy so mixed precision stays
        # consistent after save/reload (when the global policy may differ).
        self.gamma = layers.Dense(channels, kernel_initializer=init, dtype=self.dtype_policy, name="gamma")
        self.beta = layers.Dense(channels, kernel_initializer=init, dtype=self.dtype_policy, name="beta")
        # Build sub-layers explicitly so their variables exist at save/restore time
        # (Keras 3 won't auto-build children of a custom layer during deserialisation).
        self.gamma.build(cond_shape)
        self.beta.build(cond_shape)
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        # Cast modulation params to x's dtype so the affine op is dtype-consistent
        # even if the sublayer policy and x differ (robust under mixed precision).
        gamma = tf.cast(self.gamma(cond), x.dtype)[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        beta = tf.cast(self.beta(cond), x.dtype)[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        return x * (1.0 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]
