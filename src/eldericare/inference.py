"""Inference utilities for elderIcare."""

import torch

from eldericare.model import AcousticEventClassifier
from eldericare.audio import load_wav
from eldericare.features import extract_features


CLASS_NAMES = {
    0: "background",
    1: "speech",
    2: "impact",
}


def predict_event(
    model: AcousticEventClassifier,
    features: torch.Tensor,
) -> tuple[str, float]:
    """Predict an acoustic event from a feature vector.

    Parameters
    ----------
    model:
        Trained acoustic event classifier.
    features:
        One feature vector containing RMS energy and peak amplitude.

    Returns
    -------
    tuple[str, float]
        Predicted class name and confidence score.
    """
    if features.ndim == 1:
        features = features.unsqueeze(0)

    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            "Features must have shape (2,) or (batch_size, 2)."
        )

    model.eval()

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(
            probabilities.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    return CLASS_NAMES[predicted_class], confidence

def predict_wav(
    model: AcousticEventClassifier,
    path: str,
) -> tuple[str, float]:
    """Predict an acoustic event directly from a WAV file."""
    _, audio = load_wav(path)

    features = torch.from_numpy(
        extract_features(audio)
    )

    return predict_event(
        model,
        features,
    )