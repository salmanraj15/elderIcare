"""Tests for elderIcare audio loading."""

import numpy as np
from scipy.io import wavfile

from eldericare.audio import load_wav


def test_load_wav(tmp_path):
    """A valid WAV file should load with its sample rate and samples."""

    sample_rate = 16_000
    samples = np.array([0, 1000, -1000, 500, -500], dtype=np.int16)

    wav_path = tmp_path / "test.wav"
    wavfile.write(wav_path, sample_rate, samples)

    loaded_rate, loaded_samples = load_wav(wav_path)

    assert loaded_rate == sample_rate
    np.testing.assert_array_equal(loaded_samples, samples)
