# Contributing to elderIcare

Thank you for contributing to elderIcare.

The project is currently in experimental development. Contributions should prioritize correctness, reproducibility, testing, privacy, and clear documentation.

## Development Setup

Clone the repository and create a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

## Project Structure

The main source code is located in:

```text
src/eldericare/
```

Tests are located in:

```text
tests/
```

Documentation is located in:

```text
docs/
```

Experimental work is documented in:

```text
experiments/
```

Command-line utilities are located in:

```text
scripts/
```

## Running Tests

Run the complete test suite with:

```powershell
python -m pytest -v
```

All tests should pass before changes are integrated into the main workflow.

## Development Workflow

Development should proceed incrementally.

For a new capability:

1. Define the intended behavior.
2. Implement the smallest useful change.
3. Add or update automated tests.
4. Run the complete test suite.
5. Update relevant documentation.
6. Review the change for privacy and safety implications.
7. Commit the change using the project's commit message convention.

## Testing Requirements

New functionality should have automated tests where practical.

Tests should verify:

- Expected behavior
- Invalid inputs
- Important edge cases
- Error handling
- Compatibility with existing functionality

Changes should not intentionally reduce existing test coverage without a documented reason.

## Documentation

Documentation should be updated when functionality changes.

Relevant documentation may include:

- `README.md`
- `docs/architecture.md`
- `docs/dataset.md`
- `docs/model.md`
- `docs/privacy.md`
- `docs/raspberry-pi.md`
- `docs/training.md`
- Experiment documentation in `experiments/`

Documentation should clearly distinguish between:

- Implemented functionality
- Experimental functionality
- Planned functionality
- Known limitations

Do not document planned functionality as if it were already implemented.

## Commit Messages

Use a short, lowercase prefix describing the type of change.

Examples:

```text
feature: add model evaluation
fix: correct dataset split
test: add inference coverage
docs: update training documentation
refactor: simplify feature extraction
chore: update dependencies
```

Keep commit messages concise and focused on the main change.

## Pull Requests

A pull request should describe:

- What changed
- Why the change was made
- How the change was tested
- Any known limitations
- Any relevant documentation changes

Before submitting a pull request, run:

```powershell
python -m pytest -v
```

Include the test result in the pull request description when appropriate.

## Dataset Contributions

Audio data requires additional care.

Before adding recordings:

- Confirm that the project has appropriate rights and consent.
- Document recording provenance.
- Avoid unnecessary personally identifying information.
- Follow the guidance in `docs/dataset.md`.
- Follow the privacy principles in `docs/privacy.md`.
- Do not commit private recordings to the repository.

The current synthetic dataset is intended for development and testing and should not be treated as representative of real-world acoustic environments.

## Model Contributions

Changes to the model should include appropriate tests.

When changing the model architecture, document:

- Input features
- Output classes
- Architecture changes
- Training configuration
- Evaluation results
- Known limitations

Synthetic-dataset performance must not be presented as evidence of real-world performance.

## Safety

elderIcare is an experimental acoustic event detection project.

Contributors should avoid making unsupported claims about health, safety, or medical conditions.

For example, an `impact` classification should not automatically be described as proof that a person has fallen.

Acoustic classification should remain distinct from interpretation of the underlying real-world event.

## Privacy

Contributors should follow a privacy-by-design approach.

Avoid:

- Unnecessary collection of personal information
- Unnecessary storage of raw audio
- Committing private recordings to Git
- Publishing recordings without appropriate authorization
- Including unnecessary identifying information in metadata

See `docs/privacy.md` for additional guidance.

## Experimental Work

Experiments should be documented in:

```text
experiments/
```

Experiment documentation should describe the purpose, approach, results, and limitations.

Experimental results should not be treated as production validation unless supported by appropriate testing.

## Code Quality

Prefer:

- Small, focused changes
- Clear function and variable names
- Type hints where appropriate
- Useful docstrings
- Deterministic behavior where practical
- Explicit error handling
- Automated tests

Avoid unnecessary complexity.

## Scope

Contributions should focus on functionality that supports the project's acoustic event detection goals.

New functionality should be introduced incrementally and should include appropriate tests and documentation before being integrated into the main workflow.