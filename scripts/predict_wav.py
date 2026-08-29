"""Predict an acoustic event from a WAV file."""

import sys
from pathlib import Path

from eldericare.inference import predict_wav
from eldericare.model import load_model


def main() -> None:
    """Run WAV event prediction from the command line."""
    if len(sys.argv) != 2:
        print("Usage: python scripts\\predict_wav.py <wav-file>")
        raise SystemExit(1)

    wav_path = Path(sys.argv[1])
    model_path = Path("models") / "baseline.pt"

    if not wav_path.exists():
        print(f"Error: WAV file not found: {wav_path}")
        raise SystemExit(1)

    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        print("Run: python -m eldericare.train")
        raise SystemExit(1)

    model = load_model(model_path)

    event, confidence = predict_wav(
        model,
        wav_path,
    )

    print(f"Event: {event}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()