"""Dataset utilities for elderIcare."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def create_synthetic_dataset(
    samples_per_class: int = 100,
    seed: int = 42,
) -> TensorDataset:
    """Create a small synthetic dataset for development and testing.

    Classes:
        0: background
        1: speech
        2: impact

    This dataset is only for verifying the training pipeline.
    It is not representative of real-world acoustic data.
    """
    rng = np.random.default_rng(seed)

    background = rng.normal(
        loc=[0.05, 0.10],
        scale=[0.01, 0.02],
        size=(samples_per_class, 2),
    )

    speech = rng.normal(
        loc=[0.20, 0.35],
        scale=[0.03, 0.04],
        size=(samples_per_class, 2),
    )

    impact = rng.normal(
        loc=[0.40, 0.90],
        scale=[0.04, 0.05],
        size=(samples_per_class, 2),
    )

    features = np.vstack([background, speech, impact]).astype(np.float32)

    labels = np.concatenate(
        [
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            np.full(samples_per_class, 2),
        ]
    ).astype(np.int64)

    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels)

    return TensorDataset(feature_tensor, label_tensor)