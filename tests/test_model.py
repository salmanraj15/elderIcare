"""Tests for the elderIcare acoustic event classifier."""

import torch

from eldericare.model import AcousticEventClassifier


def test_model_output_shape():
    """The model should produce one prediction per input sample."""
    model = AcousticEventClassifier()

    features = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=torch.float32,
    )

    output = model(features)

    assert output.shape == (3, 3)


def test_model_custom_class_count():
    """The model should support a custom number of classes."""
    model = AcousticEventClassifier(
        input_features=2,
        num_classes=5,
    )

    features = torch.randn(4, 2)

    output = model(features)

    assert output.shape == (4, 5)