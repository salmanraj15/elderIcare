"""Analyze a synthetic impact-like audio signal for elderIcare."""

import numpy as np
import matplotlib.pyplot as plt

from eldericare.features import peak_amplitude, rms_energy


SAMPLE_RATE = 16_000
DURATION = 2.0
IMPACT_TIME = 1.0
IMPACT_DURATION = 0.08


def create_impact_signal() -> np.ndarray:
    """Create a synthetic impact-like audio signal."""
    total_samples = int(SAMPLE_RATE * DURATION)
    audio = np.zeros(total_samples, dtype=np.float32)

    impact_start = int(IMPACT_TIME * SAMPLE_RATE)
    impact_samples = int(IMPACT_DURATION * SAMPLE_RATE)

    impact_time = np.arange(impact_samples) / SAMPLE_RATE

    impact = (
        np.exp(-50 * impact_time)
        * np.sin(2 * np.pi * 150 * impact_time)
    )

    impact_end = impact_start + impact_samples
    audio[impact_start:impact_end] = impact

    return audio


def main() -> None:
    """Generate, analyze, and visualize the signal."""
    audio = create_impact_signal()

    rms = rms_energy(audio)
    peak = peak_amplitude(audio)

    print(f"RMS energy: {rms:.6f}")
    print(f"Peak amplitude: {peak:.6f}")

    time = np.arange(len(audio)) / SAMPLE_RATE

    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("elderIcare - Synthetic Impact Signal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()