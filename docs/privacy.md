# elderIcare Privacy

## Overview

elderIcare is an experimental acoustic event detection project.

Audio recordings may contain sensitive information, including conversations and other information that could identify individuals.

Privacy should therefore be considered throughout dataset collection, development, testing, and deployment.

## Privacy-by-Design Principle

The project should follow a privacy-by-design approach.

Privacy considerations should be incorporated into the system before collecting or processing real-world audio.

The project should collect and retain only the information necessary for the intended development or evaluation purpose.

## Audio Collection

Before recording audio containing people:

- Obtain appropriate consent.
- Clearly define the purpose of the recording.
- Record only the necessary audio.
- Avoid unnecessary recording of private conversations.
- Document the recording environment and intended use.
- Store recordings securely.

The project should not assume that an audio recording is safe to use simply because it was technically possible to record it.

## Dataset Provenance

Each recording included in a dataset should have documented provenance.

Relevant information may include:

- Recording identifier
- Recording date
- Recording environment
- Recording device
- Sampling rate
- Audio format
- Intended class
- Relevant recording notes

Only recordings for which the project has appropriate rights and consent should be included.

See `docs/dataset.md` for the dataset specification.

## Personally Identifying Information

Metadata should not contain unnecessary personally identifying information.

Where possible:

- Use recording identifiers instead of personal names.
- Avoid unnecessary addresses or exact locations.
- Avoid storing unrelated personal information.
- Keep identifying information separate from technical dataset metadata where appropriate.

## Raw Audio

Raw audio should be treated as potentially sensitive data.

The project should:

- Keep raw recordings separate from source code.
- Avoid committing private recordings to Git.
- Restrict access to recordings where appropriate.
- Store recordings securely.
- Remove recordings that are no longer required, subject to applicable retention requirements.

Private recordings should not be published as project examples.

## Development and Testing

The current development pipeline uses a synthetic dataset.

The synthetic dataset is intended for development and testing and does not represent real-world elder-care audio.

Using synthetic data where practical can reduce the need to expose real personal audio during early development.

Real-world recordings should only be introduced when there is a clear development or evaluation purpose and appropriate consent and data handling are in place.

## Sharing Data

Audio recordings should not be shared publicly unless the project has established that the recordings can be shared for the intended purpose.

Before sharing a dataset, consider:

- Consent
- Usage rights
- Personal information
- Sensitive content
- Metadata
- Storage and access controls
- Intended audience

Removing a person's name from metadata does not necessarily make an audio recording anonymous.

## Model Artifacts

Trained model files may encode information about the data used during training.

Model artifacts should therefore be handled responsibly, particularly when trained using real-world recordings.

Before distributing a model trained on sensitive data, the project should consider the privacy implications of the training data and the intended deployment environment.

## Deployment

Any future deployment involving microphones or continuous audio capture should clearly define:

- What audio is processed
- Whether audio is stored
- How long data is retained
- Where processing occurs
- Who can access stored data
- Whether audio leaves the device
- How users can understand and control the system

These requirements should be finalized before production deployment.

## Elder-Care Context

elderIcare is intended to explore acoustic event detection relevant to elder-care environments.

The project should avoid unnecessary surveillance.

The system should process only the audio required for its intended function and should avoid collecting unrelated personal information.

## Safety and Privacy Boundaries

Privacy protections do not make acoustic predictions medically reliable.

The system must not claim that an acoustic classification proves:

- A person has fallen.
- A medical emergency is occurring.
- A person requires medical treatment.

Acoustic classification and real-world interpretation should remain separate system concerns.

Any future alerting system should communicate uncertainty and, where appropriate, require human verification.

## Current Limitations

The current project does not yet provide a production privacy or data-governance system.

The current implementation does not include:

- A production audio collection service
- A production data-retention system
- A deployed access-control system
- A production privacy-management interface
- A production Raspberry Pi audio service

These areas require additional design and validation before real-world deployment.