"""Tests for the elderIcare acoustic event classifier."""

import torch

from eldericare.model import (
    AcousticEventClassifier,
    load_model,
    save_model,
)


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

def test_save_and_load_model(tmp_path):
    """A saved model should produce the same predictions after loading."""
    model = AcousticEventClassifier()

    features = torch.tensor(
        [
            [0.1, 0.2],
            [0.5, 0.8],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        original_output = model(features)

    model_path = tmp_path / "model.pt"

    save_model(model, model_path)

    loaded_model = load_model(model_path)

    with torch.no_grad():
        loaded_output = loaded_model(features)

    torch.testing.assert_close(
        original_output,
        loaded_output,
    )