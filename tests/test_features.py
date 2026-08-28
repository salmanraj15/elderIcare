"""Tests for elderIcare audio features."""

import numpy as np
import pytest

from eldericare.features import rms_energy


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