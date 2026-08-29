"""Tests for elderIcare inference utilities."""

import torch
import numpy as np
import subprocess
import sys

from eldericare.inference import predict_event, predict_wav
from eldericare.model import AcousticEventClassifier
from scipy.io import wavfile
from eldericare.dataset import create_synthetic_dataset
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

def test_predict_wav(tmp_path):
    """WAV input should produce a valid event prediction."""
    sample_rate = 16_000
    samples = np.zeros(16_000, dtype=np.int16)

    wav_path = tmp_path / "test.wav"
    wavfile.write(
        wav_path,
        sample_rate,
        samples,
    )

    model = AcousticEventClassifier()

    event, confidence = predict_wav(
        model,
        str(wav_path),
    )

    assert event in {
        "background",
        "speech",
        "impact",
    }

def test_predict_wav_cli_requires_argument():
    """CLI should reject a missing WAV argument."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/predict_wav.py",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Usage:" in result.stdout


def test_predict_wav_cli_rejects_missing_file():
    """CLI should reject a WAV file that does not exist."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/predict_wav.py",
            "missing.wav",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "WAV file not found" in result.stdout

def test_predict_wav_with_trained_model(tmp_path):
    """A trained model should predict directly from a WAV file."""
    dataset = create_synthetic_dataset(
        samples_per_class=30,
    )

    model = train_model(
        dataset,
        epochs=50,
    )

    sample_rate = 16_000
    samples = np.zeros(
        sample_rate,
        dtype=np.int16,
    )

    wav_path = tmp_path / "trained_test.wav"

    wavfile.write(
        wav_path,
        sample_rate,
        samples,
    )

    event, confidence = predict_wav(
        model,
        str(wav_path),
    )

    assert event in {
        "background",
        "speech",
        "impact",
    }

    assert 0.0 <= confidence <= 1.0

def test_predict_wav_returns_event_and_confidence(tmp_path):
    """WAV inference should return a valid event and confidence."""
    import numpy as np
    from scipy.io import wavfile

    from eldericare.dataset import CLASS_NAMES
    from eldericare.inference import predict_wav
    from eldericare.model import AcousticEventClassifier

    sample_rate = 16000

    audio = np.zeros(
        sample_rate,
        dtype=np.float32,
    )

    wav_path = tmp_path / "test.wav"

    wavfile.write(
        wav_path,
        sample_rate,
        audio,
    )

    model = AcousticEventClassifier()

    event, confidence = predict_wav(
        model,
        wav_path,
    )

    assert event in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0