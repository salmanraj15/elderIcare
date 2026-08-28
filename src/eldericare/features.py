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