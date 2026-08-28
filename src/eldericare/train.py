"""Training utilities for elderIcare."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eldericare.model import AcousticEventClassifier

torch.manual_seed(42)

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

if __name__ == "__main__":
    from pathlib import Path

    from eldericare.dataset import create_synthetic_dataset
    from eldericare.model import save_model

    dataset = create_synthetic_dataset(
        samples_per_class=100,
        seed=42,
    )

    model = train_model(
        dataset,
        epochs=50,
    )

    output_path = Path("models") / "baseline.pt"

    save_model(
        model,
        output_path,
    )

    print(f"Model saved to: {output_path}")

def evaluate_model(
    model: AcousticEventClassifier,
    dataset: TensorDataset,
) -> float:
    """Evaluate classification accuracy on a dataset.

    This function does not update model parameters.
    """
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)

            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return correct / total