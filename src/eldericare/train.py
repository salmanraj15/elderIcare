"""Training utilities for elderIcare."""

from pathlib import Path

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
    """Train the acoustic event classifier."""
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


def evaluate_model(
    model: AcousticEventClassifier,
    dataset: TensorDataset,
) -> float:
    """Evaluate classification accuracy on a dataset."""
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

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return correct / total

def evaluate_model_per_class(
    model: AcousticEventClassifier,
    dataset: TensorDataset,
    num_classes: int = 3,
) -> dict[int, float]:
    """Evaluate accuracy separately for each class."""
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model.eval()

    correct = [0] * num_classes
    total = [0] * num_classes

    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)

            for label, prediction in zip(labels, predictions):
                class_index = int(label.item())

                if prediction.item() == class_index:
                    correct[class_index] += 1

                total[class_index] += 1

    return {
        class_index: (
            correct[class_index] / total[class_index]
            if total[class_index] > 0
            else 0.0
        )
        for class_index in range(num_classes)
    }

if __name__ == "__main__":
    from eldericare.dataset import (
        create_synthetic_dataset,
        split_dataset,
    )
    from eldericare.model import save_model

    dataset = create_synthetic_dataset(
        samples_per_class=100,
        seed=42,
    )

    training_dataset, evaluation_dataset = split_dataset(
        dataset,
        evaluation_fraction=0.2,
        seed=42,
    )

    model = train_model(
        training_dataset,
        epochs=50,
    )

    accuracy = evaluate_model(
        model,
        evaluation_dataset,
    )

    output_path = Path("models") / "baseline.pt"

    save_model(
        model,
        output_path,
    )

    print(f"Evaluation accuracy: {accuracy:.2%}")
    print(f"Model saved to: {output_path}")