"""Tests for elderIcare inference utilities."""

import torch

from eldericare.dataset import create_synthetic_dataset
from eldericare.inference import predict_event
from eldericare.train import train_model


def test_predict_event_returns_valid_class():
    """Inference should return a known class and confidence."""
    dataset = create_synthetic_dataset(
        samples_per_class=20,
    )

    model = train_model(
        dataset,
        epochs=30,
    )

    features, _ = dataset[0]

    event, confidence = predict_event(
        model,
        features,
    )

    assert event in {
        "background",
        "speech",
        "impact",
    }

    assert 0.0 <= confidence <= 1.0


def test_predict_event_rejects_wrong_feature_shape():
    """Inference should reject feature vectors with the wrong size."""
    dataset = create_synthetic_dataset(
        samples_per_class=5,
    )

    model = train_model(
        dataset,
        epochs=2,
    )

    invalid_features = torch.tensor(
        [0.1, 0.2, 0.3],
        dtype=torch.float32,
    )

    try:
        predict_event(model, invalid_features)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid feature shape."
        )