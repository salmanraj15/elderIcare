"""Audio loading utilities for elderIcare."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file.

    Parameters
    ----------
    path:
        Path to a WAV audio file.

    Returns
    -------
    tuple[int, np.ndarray]
        Sample rate and audio samples.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the file is not a WAV file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a WAV file, got: {path.suffix}")

    sample_rate, audio = wavfile.read(path)

    return sample_rate, np.asarray(audio)