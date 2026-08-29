# Security Policy

## Overview

elderIcare is an experimental acoustic event detection project.

The project may eventually process audio recordings and other data that could contain sensitive information. Security and privacy should therefore be considered throughout development.

## Supported Versions

The project is currently under active experimental development.

There are no formal production releases or long-term supported versions at this time.

Security fixes should be applied to the current development version where practical.

## Reporting a Security Issue

If you discover a potential security vulnerability, do not publicly disclose sensitive details before the issue has been assessed.

Report the issue privately to the project maintainers.

When reporting a security issue, provide:

- A description of the issue
- Steps to reproduce the issue, where possible
- The affected component or file
- The potential impact
- Any relevant logs or supporting information

Do not include passwords, private audio recordings, personal information, or other sensitive data in a security report unless it is strictly necessary.

## Audio and Personal Data

Audio recordings may contain personal or sensitive information.

Contributors should:

- Obtain appropriate consent before recording people.
- Collect only audio that is necessary for the intended purpose.
- Avoid unnecessary recording of private conversations.
- Store recordings securely.
- Avoid committing raw recordings to the Git repository.
- Remove unnecessary identifying information from metadata.
- Do not publicly distribute private recordings without appropriate authorization.

See `docs/privacy.md` and `docs/dataset.md` for additional guidance.

## Repository Security

Contributors should avoid committing sensitive information to the repository.

Do not commit:

- Passwords
- API keys
- Access tokens
- Private credentials
- Private audio recordings
- Personally identifying information
- Other confidential information

Use environment variables or appropriate secret-management mechanisms for credentials when they are required.

## Model and Dataset Security

Model files and datasets should be treated as project artifacts.

Dataset provenance should be documented.

Changes to datasets and models should be traceable and documented where practical.

The project should not use unverified third-party data or artifacts without understanding their provenance and applicable usage rights.

## Dependency Security

Project dependencies should be kept reasonably up to date.

When dependencies are added or changed:

- Update the appropriate dependency files.
- Run the automated test suite.
- Review the dependency's purpose and source.
- Avoid unnecessary dependencies.

## Safety Boundary

Security and safety are related but distinct concerns.

elderIcare is an experimental acoustic-event detection system and should not make unsupported claims about medical conditions or emergencies.

For example:

- An `impact` classification does not prove that a person has fallen.
- A `help_call` classification does not prove that a person is in danger.
- An acoustic classification does not establish that medical treatment is required.

Any future alerting system should account for uncertainty and, where appropriate, require human verification.

## Disclosure

Confirmed security issues should be handled responsibly.

After an issue has been assessed and addressed, the project maintainers may document the issue and its resolution without exposing sensitive information.