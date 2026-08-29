# Model

## Acoustic Event Classifier

The current model is `AcousticEventClassifier`, implemented in:

```text
src/eldericare/model.py
```

`AcousticEventClassifier` is a small feed-forward neural network implemented using PyTorch.

## Input Features

The model currently accepts two numerical audio features:

- RMS energy
- Peak amplitude

The default input size is therefore:

```text
2 features
```

## Output Classes

The model currently produces three class outputs:

- `background`
- `speech`
- `impact`

The default number of classes is:

```text
3 classes
```

## Network Architecture

The current network consists of:

```text
Input: 2 features
    ↓
Linear(2 → 8)
    ↓
ReLU
    ↓
Linear(8 → 4)
    ↓
ReLU
    ↓
Linear(4 → 3)
    ↓
Output: 3 class logits
```

The model returns logits. Class probabilities are calculated during inference using softmax.

## Model Configuration

`AcousticEventClassifier` supports configurable input and output sizes.

The current defaults are:

- Input features: `2`
- Hidden layer 1: `8` units
- Hidden layer 2: `4` units
- Output classes: `3`

## Model Saving and Loading

The model module provides:

- `save_model()` for saving model parameters.
- `load_model()` for loading model parameters.

The baseline model is saved locally as:

```text
models/baseline.pt
```

Model artifacts are excluded from Git using:

```text
*.pt
*.pth
*.onnx
*.tflite
```

## Training

The model is trained by `train_model()` in:

```text
src/eldericare/train.py
```

The current training configuration uses:

- PyTorch
- Cross-entropy loss
- Adam optimizer
- Learning rate: `0.01`
- Batch size: `32`
- Epochs: `50`

## Evaluation

The training module currently supports:

- Overall accuracy
- Per-class accuracy

Evaluation is performed using a separate evaluation dataset created by the dataset splitting utilities.

## Limitations

The current model is a development baseline.

It has been tested using the synthetic development dataset, but performance on real-world acoustic recordings has not been established.

The model currently uses only two simple audio features and three acoustic event classes.

The model should therefore not be considered a production-ready elder-care event detection system.