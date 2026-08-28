"""Baseline neural network model for elderIcare."""

import torch
from torch import nn


class AcousticEventClassifier(nn.Module):
    """Small feed-forward classifier for acoustic event features."""

    def __init__(
        self,
        input_features: int = 2,
        num_classes: int = 3,
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