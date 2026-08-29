# elderIcare Architecture

## Overview

elderIcare is currently a small acoustic event classification pipeline.

The current development pipeline is:

WAV audio

→ audio loading

→ feature extraction

→ acoustic event classifier

→ event prediction

The current event classes are:

- `background`

- `speech`

- `impact`

The current model uses two numerical audio features:

- RMS energy

- Peak amplitude

## Project Structure

```text

src/eldericare/

├── audio.py

├── dataset.py

├── features.py

├── inference.py

├── model.py

└── train.py

scripts/

├── predict_wav.py

└── visualize_audio.py

tests/

├── test_audio.py

├── test_dataset.py

├── test_features.py

├── test_inference.py

└── test_model.py

```

## Components

### `audio.py`

Responsible for loading WAV audio files and returning the sample rate and audio samples.

### `features.py`

Provides audio preprocessing and feature extraction.

#### Current Features

- RMS energy

- Peak amplitude

It also supports splitting audio into fixed-duration windows.

### `dataset.py`

- Provides the synthetic development dataset used to verify the training pipeline.

- Provides a deterministic, stratified train/evaluation split.

- The synthetic dataset is explicitly intended for development and testing. It is not representative of real-world acoustic data.

### `model.py`

- Contains `AcousticEventClassifier`, a small feed-forward neural network.

- The current architecture accepts two input features and produces three class outputs.

- Provides model save/load functions.

### `train.py`

Provides model training and evaluation utilities.

#### Training Uses

- PyTorch

- Cross-entropy loss

- Adam optimizer

#### Evaluation Reports

- Overall accuracy

- Per-class accuracy

The training module can also train and save the baseline model.

### `inference.py`

Provides prediction utilities.

#### `predict_event()`

Accepts a feature vector and returns:

- Predicted event

- Confidence

#### `predict_wav()`

Connects WAV loading and feature extraction to the prediction function.

### `scripts/predict_wav.py`

Provides a command-line interface for predicting an event from a WAV file.

#### Example

```powershell

python scripts\predict_wav.py path\to\audio.wav

```

## Current Limitations

The current classifier is a development baseline.

The synthetic dataset is not representative of real-world elder-care acoustic environments.

The current feature representation contains only two simple numerical features.

Real-world performance has not yet been established.

The project does not yet contain a production audio dataset or a deployed Raspberry Pi inference service.

## Development Principle

The project should continue to add functionality incrementally.

Each new capability should have automated tests before it is integrated into the main workflow.