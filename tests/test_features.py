"""Tests for elderIcare audio features."""

import numpy as np
import pytest

from eldericare.features import (
    extract_features,
    extract_window_features,
    normalize_audio,
    peak_amplitude,
    rms_energy,
    window_audio,
)


def test_rms_energy():
    """RMS energy should be calculated correctly."""
    audio = np.array([1.0, -1.0, 1.0, -1.0])

    result = rms_energy(audio)

    assert result == pytest.approx(1.0)


def test_rms_energy_empty_signal():
    """Empty audio should raise a ValueError."""
    audio = np.array([], dtype=np.float32)

    with pytest.raises(ValueError):
        rms_energy(audio)

def test_normalize_audio():
    """Audio should be normalized to a maximum absolute value of 1."""
    audio = np.array([0.0, 2.0, -4.0, 1.0])

    result = normalize_audio(audio)

    np.testing.assert_allclose(result, [0.0, 0.5, -1.0, 0.25])


def test_normalize_silent_audio():
    """Silent audio should remain silent."""
    audio = np.zeros(4, dtype=np.float32)

    result = normalize_audio(audio)

    np.testing.assert_array_equal(result, audio)

def test_peak_amplitude():
    """Peak amplitude should return the largest absolute sample."""
    audio = np.array([0.2, -0.8, 0.4, 0.6])

    result = peak_amplitude(audio)

    assert result == pytest.approx(0.8)

def test_window_audio():
    """Audio should be divided into fixed-duration windows."""
    sample_rate = 10
    audio = np.arange(25)

    windows = window_audio(
        audio,
        sample_rate=sample_rate,
        window_duration=1.0,
    )

    assert len(windows) == 3

    np.testing.assert_array_equal(windows[0], np.arange(10))
    np.testing.assert_array_equal(windows[1], np.arange(10, 20))
    np.testing.assert_array_equal(windows[2], np.arange(20, 25))


def test_window_audio_invalid_duration():
    """A non-positive window duration should raise a ValueError."""
    audio = np.arange(10)

    with pytest.raises(ValueError):
        window_audio(audio, sample_rate=10, window_duration=0)

def test_extract_features():
    """Feature extraction should return RMS energy and peak amplitude."""
    audio = np.array([1.0, -1.0, 0.5, -0.5])

    features = extract_features(audio)

    assert features.shape == (2,)
    assert features.dtype == np.float32
    assert features[0] == pytest.approx(np.sqrt(0.625))
    assert features[1] == pytest.approx(1.0)

def test_extract_window_features():
    """Each audio window should produce one feature vector."""
    sample_rate = 10
    audio = np.array(
        [
            1.0, -1.0, 1.0, -1.0, 0.0,
            2.0, -2.0, 2.0, -2.0, 0.0,
        ]
    )

    features = extract_window_features(
        audio,
        sample_rate=sample_rate,
        window_duration=0.5,
    )

    assert features.shape == (2, 2)

    assert features[0, 0] == pytest.approx(np.sqrt(0.8))
    assert features[0, 1] == pytest.approx(1.0)

    assert features[1, 0] == pytest.approx(np.sqrt(3.2))
    assert features[1, 1] == pytest.approx(2.0)