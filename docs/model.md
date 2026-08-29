# elderIcare Model

## Acoustic Event Classifier

The current model is `AcousticEventClassifier`, implemented in:

```text
src/eldericare/model.py
```

It is a small feed-forward neural network implemented using PyTorch.

The current model is a development baseline and is intended to verify the training and inference pipeline.

## Input Features

The model currently accepts two numerical audio features:

- RMS energy
- Peak amplitude

The expected input shape for a single feature vector is:

```text
(2,)
```

For a batch of samples:

```text
(batch_size, 2)
```

## Output Classes

The current baseline model supports three acoustic event classes:

- `background`
- `speech`
- `impact`

The model produces three output logits.

The planned dataset specification contains additional classes:

- `help_call`
- `cough`
- `alarm`

These classes are not currently supported by the baseline model.

## Architecture

The current network architecture is:

```text
Input
2 features
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
Output
3 class logits
```

The implementation uses:

- `torch.nn.Linear`
- `torch.nn.ReLU`

No convolutional or recurrent layers are currently used.

## Training

The model is trained using the utilities in:

```text
src/eldericare/train.py
```

The current training configuration uses:

- Cross-entropy loss
- Adam optimizer
- Learning rate: `0.01`
- Batch size: `32`
- Default training epochs: `50`

The current development dataset is synthetic.

## Evaluation

The training pipeline currently reports:

- Overall accuracy
- Per-class accuracy

Evaluation is performed using a separate evaluation split created by the dataset utilities.

High accuracy on the synthetic dataset should not be interpreted as evidence of real-world performance.

## Model Saving and Loading

The model module provides:

```text
save_model()
```

and:

```text
load_model()
```

Models are saved using PyTorch state dictionaries.

The current baseline model is saved as:

```text
models/baseline.pt
```

The model artifact is a development artifact and is not evidence of production readiness.

## Inference

Inference utilities are implemented in:

```text
src/eldericare/inference.py
```

The `predict_event()` function accepts a feature vector and returns:

- Predicted event
- Confidence score

The `predict_wav()` function connects WAV loading, feature extraction, and model inference.

## Current Limitations

The current model has important limitations:

- It uses only two simple numerical features.
- It is trained using a synthetic development dataset.
- It supports only three classes.
- It has not been validated on a production audio dataset.
- Real-world performance has not been established.
- The model does not determine whether a medical emergency has occurred.
- The model does not independently determine whether a person has fallen.

## Future Development

Potential future model improvements include:

- Supporting the planned six-class dataset
- Additional acoustic features
- Data augmentation
- More representative training data
- Evaluation across different speakers and environments
- Comparison with alternative model architectures
- Evaluation on Raspberry Pi hardware

Future changes should be introduced incrementally and supported by automated tests.