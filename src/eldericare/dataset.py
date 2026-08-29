"""Dataset utilities for elderIcare."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def create_synthetic_dataset(
    samples_per_class: int = 100,
    seed: int = 42,
) -> TensorDataset:
    """Create a small synthetic dataset for development and testing.

    Classes:
        0: background
        1: speech
        2: impact

    This dataset is only for verifying the training pipeline.
    It is not representative of real-world acoustic data.
    """
    rng = np.random.default_rng(seed)

    background = rng.normal(
        loc=[0.05, 0.10],
        scale=[0.01, 0.02],
        size=(samples_per_class, 2),
    )

    speech = rng.normal(
        loc=[0.20, 0.35],
        scale=[0.03, 0.04],
        size=(samples_per_class, 2),
    )

    impact = rng.normal(
        loc=[0.40, 0.90],
        scale=[0.04, 0.05],
        size=(samples_per_class, 2),
    )

    features = np.vstack([background, speech, impact]).astype(np.float32)

    labels = np.concatenate(
        [
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            np.full(samples_per_class, 2),
        ]
    ).astype(np.int64)

    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels)

    return TensorDataset(feature_tensor, label_tensor)

def load_dataset_manifest(
    path: str | Path,
) -> list[dict[str, str]]:
    """Load dataset recording metadata from a CSV manifest."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Dataset manifest must contain a header row."
            )

        required_fields = {
            "recording_id",
            "class",
            "path",
        }

        missing_fields = required_fields - set(reader.fieldnames)

        if missing_fields:
            raise ValueError(
                "Dataset manifest is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        return list(reader)

class IndexedTensorDataset(TensorDataset):
    """TensorDataset that keeps the original sample indices."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        indices: list[int],
    ) -> None:
        super().__init__(features, labels)
        self.indices = indices


def split_dataset(
    dataset: TensorDataset,
    evaluation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Split a dataset into stratified training and evaluation sets."""
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError(
            "evaluation_fraction must be between 0 and 1."
        )

    features, labels = dataset.tensors

    rng = np.random.default_rng(seed)

    training_indices: list[int] = []
    evaluation_indices: list[int] = []

    for class_index in torch.unique(labels).tolist():
        class_indices = torch.where(
            labels == class_index
        )[0].tolist()

        rng.shuffle(class_indices)

        evaluation_size = int(
            len(class_indices) * evaluation_fraction
        )

        evaluation_indices.extend(
            class_indices[:evaluation_size]
        )

        training_indices.extend(
            class_indices[evaluation_size:]
        )

    rng.shuffle(training_indices)
    rng.shuffle(evaluation_indices)

    training_features = features[training_indices]
    training_labels = labels[training_indices]

    evaluation_features = features[evaluation_indices]
    evaluation_labels = labels[evaluation_indices]

    return (
        IndexedTensorDataset(
            training_features,
            training_labels,
            training_indices,
        ),
        IndexedTensorDataset(
            evaluation_features,
            evaluation_labels,
            evaluation_indices,
        ),
    )