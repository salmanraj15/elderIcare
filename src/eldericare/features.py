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

def window_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_duration: float,
) -> list[np.ndarray]:
    """Split audio into fixed-duration windows.

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.
    sample_rate:
        Number of samples per second.
    window_duration:
        Window length in seconds.

    Returns
    -------
    list[np.ndarray]
        List of audio windows.

    Raises
    ------
    ValueError
        If the audio is empty or the window duration is invalid.
    """
    samples = np.asarray(audio)

    if samples.size == 0:
        raise ValueError("Audio signal cannot be empty.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if window_duration <= 0:
        raise ValueError("Window duration must be positive.")

    window_size = int(sample_rate * window_duration)

    if window_size <= 0:
        raise ValueError("Window duration is too short.")

    return [
        samples[start : start + window_size]
        for start in range(0, len(samples), window_size)
        if len(samples[start : start + window_size]) > 0
    ]

def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract a feature vector from an audio window.

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.

    Returns
    -------
    np.ndarray
        Feature vector containing RMS energy and peak amplitude.
    """
    return np.array(
        [
            rms_energy(audio),
            peak_amplitude(audio),
        ],
        dtype=np.float32,
    )

def extract_window_features(
    audio: np.ndarray,
    sample_rate: int,
    window_duration: float,
) -> np.ndarray:
    """Extract feature vectors from fixed-duration audio windows.

    Parameters
    ----------
    audio:
        Audio samples as a NumPy array.
    sample_rate:
        Number of samples per second.
    window_duration:
        Window length in seconds.

    Returns
    -------
    np.ndarray
        Two-dimensional feature matrix where each row represents
        one audio window.
    """
    windows = window_audio(
        audio,
        sample_rate=sample_rate,
        window_duration=window_duration,
    )

    return np.vstack([extract_features(window) for window in windows])