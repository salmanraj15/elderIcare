"""Baseline neural network model for elderIcare."""

import torch
from torch import nn
from pathlib import Path
from eldericare.dataset import NUM_CLASSES


class AcousticEventClassifier(nn.Module):
    """Small feed-forward classifier for acoustic event features."""

    def __init__(
        self,
        input_features: int = 2,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialize the classifier.

        Parameters
        ----------
        input_features:
            Number of numerical features provided for each audio window.
        num_classes:
            Number of acoustic event classes.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the classifier on a batch of feature vectors."""
        return self.network(features)

def save_model(
    model: AcousticEventClassifier,
    path: str | Path,
) -> None:
    """Save model parameters to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    input_features: int = 2,
    num_classes: int = NUM_CLASSES,
) -> AcousticEventClassifier:
    """Load model parameters from disk."""
    path = Path(path)

    model = AcousticEventClassifier(
        input_features=input_features,
        num_classes=num_classes,
    )

    state_dict = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model