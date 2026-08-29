# elderIcare Privacy

## Overview

elderIcare is an experimental acoustic event detection project.

Audio data can contain sensitive or personally identifiable information. Privacy should therefore be considered throughout dataset collection, storage, development, testing, and deployment.

The project follows a privacy-by-design approach.

## Audio Data

Audio recordings should be collected only when there is a clear development or research purpose.

Where practical:

- Record only the minimum audio necessary.
- Avoid recording private conversations unnecessarily.
- Avoid collecting unrelated personal information.
- Obtain appropriate consent from people who are recorded.
- Keep recordings securely stored.
- Do not publish private recordings.

## Consent

Recordings should only be included in the project when the project has the appropriate rights and consent to use them.

The project should document relevant consent and recording permissions where required.

Consent information should be stored separately from the audio data when appropriate.

## Dataset Metadata

Dataset metadata should contain only information necessary for the project.

Avoid unnecessary personally identifying information such as:

- Full names
- Home addresses
- Telephone numbers
- Email addresses
- Other unnecessary personal identifiers

Recording identifiers should be used instead of personal names where possible.

Example:

```text
recording_id: REC_000001
class: background
date: YYYY-MM-DD
device: <recording device>
sampling_rate: <Hz>
channels: 1
duration_seconds: <duration>
environment: living_room
notes: quiet room with low background fan noise
```

## Raw Recordings

Raw audio recordings should be kept separate from source code.

Raw recordings should not be committed to the Git repository unless there is an explicit and documented reason to do so.

The project should use appropriate access controls and secure storage for recordings.

## Synthetic Dataset

The current development dataset is synthetic.

It is used to verify the training and evaluation pipeline and does not represent real-world elder-care acoustic environments.

The synthetic dataset should not be treated as a substitute for privacy-reviewed real-world recordings.

## Model Artifacts

Trained model artifacts may contain information derived from training data.

Model files should therefore be handled as project artifacts and should not be assumed to be automatically free of privacy concerns.

The current repository excludes model artifacts such as:

```text
*.pt
*.pth
*.onnx
*.tflite
```

## Development and Testing

Development and testing should avoid exposing sensitive recordings unnecessarily.

Where possible:

- Use synthetic data for software testing.
- Use anonymized or appropriately consented recordings for development.
- Keep test fixtures free of unnecessary personal information.
- Avoid committing private recordings to source control.

## Deployment Considerations

Any future Raspberry Pi or other edge deployment should consider privacy before collecting or processing real-world audio.

The system should process only the audio required for its intended function.

Where practical, audio processing should occur locally rather than transmitting raw recordings unnecessarily.

The exact deployment privacy architecture has not yet been finalized.

## Safety and Privacy Boundary

elderIcare is an experimental acoustic-event detection system.

Acoustic classification should not be used to make unsupported conclusions about a person's health, safety, or medical condition.

For example, the system should not claim:

- "A person has fallen."
- "A medical emergency is occurring."
- "The person requires medical treatment."

An acoustic event may provide evidence that something happened, but acoustic classification alone does not establish the underlying real-world event.

Any future alerting system should communicate uncertainty and, where appropriate, require human verification.

## Future Work

Potential privacy improvements include:

- Formalizing the consent process.
- Defining data retention periods.
- Documenting access-control procedures.
- Establishing secure storage requirements.
- Reviewing privacy requirements before real-world dataset collection.
- Defining procedures for deleting recordings and associated metadata.
- Reviewing privacy requirements before deployment in real-world elder-care environments.