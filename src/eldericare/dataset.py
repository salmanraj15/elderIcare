"""Dataset utilities for elderIcare."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split


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

def split_dataset(
    dataset: TensorDataset,
    evaluation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Split a dataset into training and evaluation subsets.

    Parameters
    ----------
    dataset:
        Dataset to split.
    evaluation_fraction:
        Fraction of samples reserved for evaluation.
    seed:
        Random seed used for the split.

    Returns
    -------
    tuple[TensorDataset, TensorDataset]
        Training dataset followed by evaluation dataset.
    """
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError(
            "evaluation_fraction must be between 0 and 1."
        )

    evaluation_size = int(len(dataset) * evaluation_fraction)
    training_size = len(dataset) - evaluation_size

    generator = torch.Generator().manual_seed(seed)

    training_dataset, evaluation_dataset = random_split(
        dataset,
        [training_size, evaluation_size],
        generator=generator,
    )

    return training_dataset, evaluation_dataset