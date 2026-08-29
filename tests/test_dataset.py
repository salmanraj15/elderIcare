"""Tests for elderIcare dataset utilities."""

import torch
import pytest

from eldericare.dataset import (
    create_synthetic_dataset,
    split_dataset,
)


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

def test_split_dataset():
    """Dataset should be split into training and evaluation sets."""
    dataset = create_synthetic_dataset(
        samples_per_class=50,
    )

    training, evaluation = split_dataset(
        dataset,
        evaluation_fraction=0.2,
    )

    assert len(training) == 120
    assert len(evaluation) == 30
    assert len(training) + len(evaluation) == len(dataset)


def test_split_dataset_is_deterministic():
    """The same seed should produce the same split."""
    dataset = create_synthetic_dataset(
        samples_per_class=20,
    )

    training_a, evaluation_a = split_dataset(
        dataset,
        evaluation_fraction=0.2,
        seed=42,
    )

    training_b, evaluation_b = split_dataset(
        dataset,
        evaluation_fraction=0.2,
        seed=42,
    )

    assert training_a.indices == training_b.indices
    assert evaluation_a.indices == evaluation_b.indices


def test_split_dataset_rejects_invalid_fraction():
    """Invalid evaluation fractions should raise ValueError."""
    dataset = create_synthetic_dataset(
        samples_per_class=10,
    )

    with pytest.raises(ValueError):
        split_dataset(dataset, evaluation_fraction=0.0)

    with pytest.raises(ValueError):
        split_dataset(dataset, evaluation_fraction=1.0)

def test_split_dataset_preserves_all_classes():
    """Both splits should contain every class."""
    dataset = create_synthetic_dataset(
        samples_per_class=50,
    )

    training, evaluation = split_dataset(
        dataset,
        evaluation_fraction=0.2,
    )

    training_labels = {
        int(label.item())
        for _, label in training
    }

    evaluation_labels = {
        int(label.item())
        for _, label in evaluation
    }

    assert training_labels == {0, 1, 2}
    assert evaluation_labels == {0, 1, 2}