"""Tests for elderIcare audio features."""

import numpy as np
import pytest

from eldericare.features import normalize_audio, rms_energy, peak_amplitude


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