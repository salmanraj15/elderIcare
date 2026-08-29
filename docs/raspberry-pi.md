# elderIcare Raspberry Pi Deployment

## Overview

Raspberry Pi deployment is a planned direction for elderIcare.

The current project contains an experimental inference pipeline, but it does not yet provide a production Raspberry Pi inference service.

The intended direction is to evaluate whether the acoustic event classifier can perform inference locally on Raspberry Pi hardware.

## Current Status

The current implementation provides:

- WAV audio loading
- Audio feature extraction
- Model inference
- WAV-based prediction
- Model saving and loading

The current implementation does not yet provide:

- Continuous microphone capture
- A production Raspberry Pi service
- Real-time event detection
- Production alerting
- A validated Raspberry Pi deployment
- Production monitoring

## Intended Pipeline

A future Raspberry Pi implementation may follow a pipeline similar to:

```text
Microphone
    ↓
Audio capture
    ↓
Audio preprocessing
    ↓
Fixed-duration windows
    ↓
Feature extraction
    ↓
Acoustic event classifier
    ↓
Event + confidence
    ↓
Optional alerting layer
```

The exact implementation should be determined through testing on the target hardware.

## Local Inference

A primary deployment goal is to perform inference locally on the Raspberry Pi.

Local processing may reduce the need to transmit raw audio to an external service.

However, the privacy and security properties of a future deployment must be evaluated before production use.

See `docs/privacy.md` for the project's privacy principles.

## Model Deployment

The current baseline model is stored as:

```text
models/baseline.pt
```

The model is currently saved using a PyTorch state dictionary.

A future Raspberry Pi deployment should evaluate:

- Model loading time
- Inference latency
- CPU usage
- Memory usage
- Storage requirements
- Power consumption
- Long-running stability

No specific hardware performance targets have been established yet.

## Audio Input

The current inference pipeline supports WAV files.

A future Raspberry Pi implementation would need to add microphone input and continuous or repeated audio capture.

Important considerations include:

- Sampling rate
- Number of audio channels
- Audio format
- Buffer size
- Window duration
- Microphone placement
- Background noise
- Recording level

These parameters should be tested on the target hardware rather than assumed to be optimal.

## Real-Time Processing

Real-time processing is a future capability.

A future implementation should process audio in bounded windows and perform inference without requiring the complete recording to be stored.

The system should measure processing latency to determine whether the selected model and hardware configuration are suitable for the intended use.

## Event Detection

The current model classifies an audio feature vector into one of three baseline classes:

- `background`
- `speech`
- `impact`

A future Raspberry Pi implementation may process consecutive audio windows and produce a sequence of predictions.

Additional logic may eventually be required to:

- Reduce duplicate predictions
- Handle uncertain predictions
- Detect persistent events
- Apply confidence thresholds
- Combine multiple observations

These behaviors are not currently implemented.

## Alerting

Alerting is a separate future system layer.

A model prediction should not automatically be treated as proof of an emergency.

For example:

```text
impact
```

does not establish:

```text
person has fallen
```

Similarly, an acoustic classification does not establish that medical assistance is required.

Any future alerting layer should account for model uncertainty and, where appropriate, require human verification.

## Privacy

A future Raspberry Pi deployment should consider privacy from the beginning.

Where practical, local processing may allow audio to be analyzed without continuously transmitting raw recordings elsewhere.

The deployment should clearly define:

- Whether audio is stored
- How long audio is retained
- Whether audio leaves the device
- Who can access stored data
- What metadata is retained
- How users are informed about audio processing

See `docs/privacy.md` for additional guidance.

## Security

A deployed Raspberry Pi should be treated as a network-connected computing device if network access is enabled.

Future deployment work should consider:

- Operating-system updates
- Dependency updates
- Authentication
- Network exposure
- Secure configuration
- Access control
- Protection of model files
- Protection of stored audio
- Secure handling of credentials

The production security architecture has not yet been defined.

## Hardware Evaluation

Before production deployment, the system should be evaluated on the intended Raspberry Pi hardware.

Evaluation should consider:

- CPU utilization
- Memory utilization
- Inference latency
- Startup time
- Continuous runtime
- Thermal behavior
- Power consumption
- Storage usage
- Audio capture reliability

Results should be documented as experiments.

## Testing

Raspberry Pi deployment should be tested separately from the core software tests.

The existing test suite should continue to run:

```powershell
python -m pytest -v
```

Hardware-specific testing should additionally verify the behavior of:

- Audio capture
- Feature extraction
- Model inference
- Repeated inference
- Error recovery
- Long-running operation

## Current Limitations

The project does not currently provide a validated Raspberry Pi deployment.

In particular:

- No continuous microphone service is implemented.
- No real-time inference service is implemented.
- No production alerting system is implemented.
- No hardware performance benchmark has been established.
- No production deployment configuration has been finalized.

Raspberry Pi deployment should therefore be considered future experimental work.

## Future Work

Potential future Raspberry Pi work includes:

- Select target Raspberry Pi hardware.
- Implement microphone capture.
- Connect live audio to the existing feature pipeline.
- Measure inference latency.
- Measure CPU and memory usage.
- Evaluate continuous operation.
- Investigate model optimization if required.
- Define secure deployment configuration.
- Evaluate privacy implications.
- Document hardware experiments.
- Develop and test an appropriate alerting layer.

All future deployment claims should be supported by measurements from the actual target hardware.