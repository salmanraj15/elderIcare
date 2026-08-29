# Changelog

All notable changes to the elderIcare project are documented in this file.

The project is currently in experimental development.

## [Unreleased]

### Added

- Added WAV audio loading utilities.
- Added audio normalization.
- Added fixed-duration audio windowing.
- Added RMS energy feature extraction.
- Added peak amplitude feature extraction.
- Added synthetic development dataset generation.
- Added deterministic stratified train/evaluation dataset splitting.
- Added `AcousticEventClassifier`.
- Added model saving and loading utilities.
- Added model training utilities.
- Added overall accuracy evaluation.
- Added per-class accuracy evaluation.
- Added acoustic event inference utilities.
- Added WAV-based inference.
- Added command-line WAV prediction.
- Added automated tests for the core pipeline.
- Added project architecture documentation.
- Added dataset documentation.
- Added model documentation.
- Added training documentation.
- Added privacy documentation.
- Added Raspberry Pi deployment documentation.

### Current Baseline

The current development baseline uses:

- RMS energy
- Peak amplitude
- `background`
- `speech`
- `impact`
- PyTorch feed-forward neural network
- Cross-entropy loss
- Adam optimizer

The current model is trained and evaluated using a synthetic development dataset.

### Limitations

- The current dataset is synthetic.
- The synthetic dataset is not representative of real-world elder-care acoustic environments.
- Real-world model performance has not yet been established.
- The current model uses only two simple acoustic features.
- A production Raspberry Pi inference service has not yet been implemented.
- A production alerting system has not yet been implemented.

## Versioning

Dataset and experiment versions are documented separately from software releases.

Future software releases should document:

- New functionality
- Changed functionality
- Bug fixes
- Test changes
- Model or dataset changes
- Important compatibility changes