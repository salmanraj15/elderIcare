"""Tests for elderIcare dataset utilities."""

import torch
import pytest
import numpy as np

from eldericare.dataset import (
    create_synthetic_dataset,
    load_dataset_manifest,
    load_audio_dataset,
    split_dataset
)


def test_create_synthetic_dataset():
    """Synthetic dataset should contain features and labels."""
    dataset = create_synthetic_dataset(samples_per_class=10)

    assert len(dataset) == 30

    features, labels = dataset[0]

    assert features.shape == (2,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_synthetic_dataset_has_three_classes():
    """Synthetic dataset should contain three classes."""
    dataset = create_synthetic_dataset(samples_per_class=10)

    labels = torch.stack([item[1] for item in dataset])

    assert set(labels.tolist()) == {0, 1, 2}

def test_split_dataset():
    """Dataset should be split into training and evaluation sets."""
    dataset = create_synthetic_dataset(
        samples_per_class=50,
    )

    training, evaluation = split_dataset(
        dataset,
        evaluation_fraction=0.2,
    )

    assert len(training) == 120
    assert len(evaluation) == 30
    assert len(training) + len(evaluation) == len(dataset)


def test_split_dataset_is_deterministic():
    """The same seed should produce the same split."""
    dataset = create_synthetic_dataset(
        samples_per_class=20,
    )

    training_a, evaluation_a = split_dataset(
        dataset,
        evaluation_fraction=0.2,
        seed=42,
    )

    training_b, evaluation_b = split_dataset(
        dataset,
        evaluation_fraction=0.2,
        seed=42,
    )

    assert training_a.indices == training_b.indices
    assert evaluation_a.indices == evaluation_b.indices


def test_split_dataset_rejects_invalid_fraction():
    """Invalid evaluation fractions should raise ValueError."""
    dataset = create_synthetic_dataset(
        samples_per_class=10,
    )

    with pytest.raises(ValueError):
        split_dataset(dataset, evaluation_fraction=0.0)

    with pytest.raises(ValueError):
        split_dataset(dataset, evaluation_fraction=1.0)

def test_split_dataset_preserves_all_classes():
    """Both splits should contain every class."""
    dataset = create_synthetic_dataset(
        samples_per_class=50,
    )

    training, evaluation = split_dataset(
        dataset,
        evaluation_fraction=0.2,
    )

    training_labels = {
        int(label.item())
        for _, label in training
    }

    evaluation_labels = {
        int(label.item())
        for _, label in evaluation
    }

    assert training_labels == {0, 1, 2}
    assert evaluation_labels == {0, 1, 2}

def test_load_dataset_manifest(tmp_path):
    """Dataset manifest should load recording metadata."""
    manifest_path = tmp_path / "metadata.csv"

    manifest_path.write_text(
        "recording_id,class,path\n"
        "REC_000001,background,a.wav\n"
        "REC_000002,speech,b.wav\n",
        encoding="utf-8",
    )

    records = load_dataset_manifest(manifest_path)

    assert len(records) == 2
    assert records[0]["recording_id"] == "REC_000001"
    assert records[0]["class"] == "background"
    assert records[1]["path"] == "b.wav"    
    
def test_load_dataset_manifest_rejects_missing_file(tmp_path):
    """A missing dataset manifest should raise FileNotFoundError."""
    manifest_path = tmp_path / "missing.csv"

    try:
        load_dataset_manifest(manifest_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError for missing manifest."
        )


def test_load_dataset_manifest_rejects_missing_fields(tmp_path):
    """A manifest missing required fields should raise ValueError."""
    manifest_path = tmp_path / "metadata.csv"

    manifest_path.write_text(
        "recording_id,path\n"
        "REC_000001,a.wav\n",
        encoding="utf-8",
    )

    try:
        load_dataset_manifest(manifest_path)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for missing manifest fields."
        )

def test_load_audio_dataset(tmp_path):
    """Audio recordings in a manifest should become a TensorDataset."""
    from scipy.io import wavfile

    manifest_path = tmp_path / "metadata.csv"
    audio_path = tmp_path / "test.wav"

    sample_rate = 16_000
    samples = np.zeros(
        sample_rate,
        dtype=np.int16,
    )

    wavfile.write(
        audio_path,
        sample_rate,
        samples,
    )

    manifest_path.write_text(
        "recording_id,class,path\n"
        "REC_000001,background,test.wav\n",
        encoding="utf-8",
    )

    dataset = load_audio_dataset(
        manifest_path
    )

    assert len(dataset) == 1

    features, label = dataset[0]

    assert features.shape == (2,)
    assert label.item() == 0

def test_dataset_class_mapping():
    """Dataset classes should have stable numeric indices."""
    from eldericare.dataset import (
        CLASS_NAMES,
        CLASS_TO_INDEX,
        INDEX_TO_CLASS,
    )

    assert len(CLASS_NAMES) == 6

    assert CLASS_TO_INDEX["background"] == 0
    assert CLASS_TO_INDEX["speech"] == 1
    assert CLASS_TO_INDEX["help_call"] == 2
    assert CLASS_TO_INDEX["impact"] == 3
    assert CLASS_TO_INDEX["cough"] == 4
    assert CLASS_TO_INDEX["alarm"] == 5

    for index, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_INDEX[name] == index
        assert INDEX_TO_CLASS[index] == name