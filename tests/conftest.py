"""Shared pytest fixtures. Tests run on CPU with tiny tensors so the whole suite
finishes in seconds."""

from __future__ import annotations

import pytest

from cosmomap.config import Config


@pytest.fixture
def tiny_config() -> Config:
    """A minimal config: 16^3 boxes, 8^3 patches, a 2-level net."""
    c = Config()
    c.data.resolution = 16
    c.data.patch_size = 8
    c.data.redshifts = [0.0, 1.0]
    c.data.train_val_test_split = [0.6, 0.2, 0.2]
    c.model.base_filters = 8
    c.model.depth = 2
    c.model.groups = 2
    c.model.cond_hidden = 16
    c.train.batch_size = 2
    c.train.epochs = 1
    c.train.mixed_precision = False
    return c
