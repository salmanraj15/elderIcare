"""Training utilities for elderIcare."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eldericare.model import AcousticEventClassifier


def train_model(
    dataset: TensorDataset,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> AcousticEventClassifier:
    """Train the acoustic event classifier.

    This training function is intended for development and testing.
    """
    model = AcousticEventClassifier()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()

            predictions = model(features)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

    return model