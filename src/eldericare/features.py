"""Audio feature extraction utilities for elderIcare."""

import numpy as np


def rms_energy(audio: np.ndarray) -> float:
    """Calculate the root mean square (RMS) energy of an audio signal.

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.

    Returns
    -------
    float
        RMS energy of the signal.
    """
    samples = np.asarray(audio, dtype=np.float32)

    if samples.size == 0:
        raise ValueError("Audio signal cannot be empty.")

    return float(np.sqrt(np.mean(samples**2)))

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio samples to the range [-1, 1].

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.

    Returns
    -------
    np.ndarray
        Normalized audio samples.
    """
    samples = np.asarray(audio, dtype=np.float32)

    if samples.size == 0:
        raise ValueError("Audio signal cannot be empty.")

    peak = np.max(np.abs(samples))

    if peak == 0:
        return samples

    return samples / peak

def peak_amplitude(audio: np.ndarray) -> float:
    """Return the maximum absolute amplitude of an audio signal.

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.

    Returns
    -------
    float
        Maximum absolute amplitude.

    Raises
    ------
    ValueError
        If the audio signal is empty.
    """
    samples = np.asarray(audio, dtype=np.float32)

    if samples.size == 0:
        raise ValueError("Audio signal cannot be empty.")

    return float(np.max(np.abs(samples)))