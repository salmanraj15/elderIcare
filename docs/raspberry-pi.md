# elderIcare Raspberry Pi Deployment

## Overview

elderIcare is intended to eventually support inference on a Raspberry Pi or similar edge device.

The current project does not yet contain a deployed Raspberry Pi inference service.

The Raspberry Pi deployment described in this document is therefore a development direction rather than a completed production deployment.

## Current Status

The current pipeline supports:

- WAV audio loading
- Audio feature extraction
- Acoustic event classification
- WAV-based inference
- Model saving and loading
- Automated testing

The current model uses:

- RMS energy
- Peak amplitude

The current development classes are:

- `background`
- `speech`
- `impact`

## Intended Edge Pipeline

The intended Raspberry Pi workflow is:

```text
Microphone
    ↓
Audio capture
    ↓
Audio preprocessing
    ↓
Fixed-duration audio window
    ↓
Feature extraction
    ↓
Acoustic event classifier
    ↓
Predicted event + confidence
    ↓
Optional alerting layer
```

The alerting layer is not currently implemented.

## Model Deployment

The trained model is currently saved as:

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

A future Raspberry Pi deployment will need a controlled method for transferring the appropriate model artifact to the device.

## Inference

The current inference utilities provide:

### `predict_event()`

Accepts a feature vector containing:

- RMS energy
- Peak amplitude

It returns:

- Predicted event
- Confidence

### `predict_wav()`

Connects WAV loading and feature extraction to the prediction function.

This provides a useful development path for testing the model before live microphone inference is implemented.

## Hardware

The specific Raspberry Pi model and microphone hardware have not yet been finalized.

The deployment should eventually document:

- Raspberry Pi model
- Operating system
- Python version
- Microphone hardware
- Audio interface
- Sampling rate
- Number of audio channels
- Audio capture configuration
- Model format

## Performance

Raspberry Pi performance has not yet been formally benchmarked.

Before deployment, the project should measure:

- Audio capture reliability
- Feature extraction latency
- Model inference latency
- Memory usage
- CPU usage
- Continuous runtime stability
- Power consumption where relevant

Performance results should be documented using the specific hardware and software configuration tested.

## Privacy Considerations

Audio collected by an edge device may contain sensitive information.

Any future Raspberry Pi deployment should:

- Process only the audio necessary for the intended function.
- Avoid unnecessary storage of raw audio.
- Avoid unnecessary transmission of raw audio.
- Use appropriate access controls.
- Follow the project's privacy and consent requirements.

See `docs/privacy.md` for the project's privacy principles.

## Safety Boundary

The Raspberry Pi system should be treated as an acoustic event detection system.

An acoustic classification does not establish what happened in the physical world.

For example:

- `impact` does not prove that a person has fallen.
- `speech` does not indicate a person's health status.
- `help_call` does not prove that an emergency is occurring.
- `cough` does not establish a medical condition.

Any future alerting system should communicate uncertainty and, where appropriate, require human verification.

## Future Work

Potential Raspberry Pi development steps include:

- Select and document Raspberry Pi hardware.
- Select and document microphone hardware.
- Implement live microphone capture.
- Implement continuous audio windowing.
- Run feature extraction on-device.
- Load the trained model on the device.
- Measure inference latency and resource usage.
- Implement controlled alerting.
- Add automated tests for the edge inference workflow.
- Evaluate reliability under realistic environmental conditions.
- Document deployment and maintenance procedures.

## Current Limitations

The Raspberry Pi deployment is not yet production-ready.

The project currently does not provide:

- A live microphone inference service
- A finalized Raspberry Pi hardware configuration
- A production deployment package
- A formal performance benchmark
- A production alerting system
- Real-world deployment validation