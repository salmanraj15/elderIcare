"""Tests for elderIcare dataset utilities."""

import torch

from eldericare.dataset import create_synthetic_dataset


def test_create_synthetic_dataset():
    """Synthetic dataset should contain features and labels."""
    dataset = create_synthetic_dataset(samples_per_class=10)

    assert len(dataset) == 30

    features, labels = dataset[0]

    assert features.shape == (2,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_synthetic_dataset_has_three_classes():
    """Synthetic dataset should contain three classes."""
    dataset = create_synthetic_dataset(samples_per_class=10)

    labels = torch.stack([item[1] for item in dataset])

    assert set(labels.tolist()) == {0, 1, 2}