"""Composite training loss.

Combines (with weights from a hyperparameter search):

* a heteroscedastic Gaussian negative log-likelihood (NLL) that lets the model
  express per-voxel aleatoric uncertainty,
* MSE, MAE and Huber terms on the predicted mean, and
* physics-informed regularisers: a positivity prior on the electron-density mean
  and a penalty on runaway predicted variance.

    L = w_nll * NLL + w_mse * MSE + w_mae * MAE + w_huber * Huber
        + lambda_pos * E[max(0, -mu_ne)] + lambda_var * E[exp(log_var)]
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from cosmomap.config import Config

_LOG_VAR_CLIP = 10.0  # keep exp(+/- log_var) finite


def _huber(error: tf.Tensor, delta: float) -> tf.Tensor:
    abs_err = tf.abs(error)
    quad = tf.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * tf.square(quad) + delta * lin


@tf.keras.utils.register_keras_serializable(package="cosmomap")
class CosmoMapLoss(keras.losses.Loss):
    """Composite heteroscedastic + physics-informed loss.

    Parameters
    ----------
    n_out:
        Number of physical target channels (``y_true`` last-dim).
    heteroscedastic:
        If ``True``, ``y_pred`` has ``2 * n_out`` channels: ``[mean, log_var]``.
    ne_index:
        Channel index of electron density within the means, used by the
        positivity prior (``None`` disables it).
    """

    def __init__(
        self,
        n_out: int,
        *,
        heteroscedastic: bool = True,
        w_nll: float = 0.408,
        w_mse: float = 0.150,
        w_mae: float = 0.134,
        w_huber: float = 0.207,
        huber_delta: float = 1.0,
        lambda_positivity: float = 0.1,
        lambda_variance: float = 0.01,
        ne_index: int | None = None,
        name: str = "cosmomap_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.n_out = n_out
        self.heteroscedastic = heteroscedastic
        self.w_nll = w_nll
        self.w_mse = w_mse
        self.w_mae = w_mae
        self.w_huber = w_huber
        self.huber_delta = huber_delta
        self.lambda_positivity = lambda_positivity
        self.lambda_variance = lambda_variance
        self.ne_index = ne_index

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.heteroscedastic:
            mean = y_pred[..., : self.n_out]
            log_var = tf.clip_by_value(y_pred[..., self.n_out :], -_LOG_VAR_CLIP, _LOG_VAR_CLIP)
        else:
            mean, log_var = y_pred, None

        error = y_true - mean
        loss = (
            self.w_mse * tf.reduce_mean(tf.square(error))
            + self.w_mae * tf.reduce_mean(tf.abs(error))
            + self.w_huber * tf.reduce_mean(_huber(error, self.huber_delta))
        )

        if log_var is not None:
            # Gaussian NLL: 0.5 * [ (y - mu)^2 / sigma^2 + log sigma^2 ]
            nll = 0.5 * tf.reduce_mean(tf.square(error) * tf.exp(-log_var) + log_var)
            loss += self.w_nll * nll
            loss += self.lambda_variance * tf.reduce_mean(tf.exp(log_var))

        if self.ne_index is not None and self.lambda_positivity > 0:
            # Positivity prior on the electron-density mean (penalise mu_ne < 0).
            mu_ne = mean[..., self.ne_index]
            loss += self.lambda_positivity * tf.reduce_mean(tf.nn.relu(-mu_ne))

        return loss

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "n_out": self.n_out,
                "heteroscedastic": self.heteroscedastic,
                "w_nll": self.w_nll,
                "w_mse": self.w_mse,
                "w_mae": self.w_mae,
                "w_huber": self.w_huber,
                "huber_delta": self.huber_delta,
                "lambda_positivity": self.lambda_positivity,
                "lambda_variance": self.lambda_variance,
                "ne_index": self.ne_index,
            }
        )
        return config


def make_loss(config: Config) -> CosmoMapLoss:
    """Build the composite loss from a :class:`~cosmomap.config.Config`."""
    fields_out = config.data.fields_out
    ne_index = fields_out.index("ne") if "ne" in fields_out else None
    lc = config.loss
    return CosmoMapLoss(
        n_out=config.n_out,
        heteroscedastic=config.model.heteroscedastic,
        w_nll=lc.w_nll,
        w_mse=lc.w_mse,
        w_mae=lc.w_mae,
        w_huber=lc.w_huber,
        huber_delta=lc.huber_delta,
        lambda_positivity=lc.lambda_positivity,
        lambda_variance=lc.lambda_variance,
        ne_index=ne_index,
    )
