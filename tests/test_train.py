"""Tests for elderIcare training utilities."""

import torch

from eldericare.dataset import create_synthetic_dataset
from eldericare.train import train_model


def test_train_model_returns_classifier():
    """Training should return a usable model."""
    dataset = create_synthetic_dataset(
        samples_per_class=10,
    )

    model = train_model(
        dataset,
        epochs=2,
    )

    features, _ = dataset[0]

    with torch.no_grad():
        output = model(features.unsqueeze(0))

    assert output.shape == (1, 3)


def test_trained_model_can_predict_synthetic_data():
    """The baseline model should learn the simple synthetic pattern."""
    dataset = create_synthetic_dataset(
        samples_per_class=30,
        seed=42,
    )

    model = train_model(
        dataset,
        epochs=50,
    )

    features = torch.stack(
        [item[0] for item in dataset]
    )

    labels = torch.tensor(
        [item[1] for item in dataset]
    )

    model.eval()

    with torch.no_grad():
        predictions = model(features).argmax(dim=1)

    accuracy = (predictions == labels).float().mean().item()

    assert accuracy >= 0.90