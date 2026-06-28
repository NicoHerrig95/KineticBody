# BodyScan

BodyScan is a Python package designed to provide functionality for body analysis and scanning workflows by extracting kinectic metrices from a moving human object. This repository contains the core package, project configuration, and supporting code for development and deployment. The package is build around the 'KineticBody' class, which computes 

## Overview

The goal of BodyScan is to provide a modular and extensible foundation for processing, analyzing, and managing body scan data.

## Features

* Python package structure using modern packaging standards
* Easy installation via `pip`
* Extensible architecture for future body analysis capabilities
* Development-friendly project layout

## Requirements

* Python 3.9+ (recommended)
* pip

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd BodyScan
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
```

or

```powershell
.venv\Scripts\activate
```

Install the package in editable mode:

```bash
pip install -e .
```

## Project Structure

```text
BodyScan/
├── pyproject.toml
├── src/
│   └── bodyscan/
├── tests/
└── README.md
```

## Usage

Import the package in Python:

```python
import bodyscan
```

Example usage will be added as the project evolves.

## Development

Install development dependencies:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```

Format code:

```bash
black .
```

Lint code:

```bash
ruff check .
```

## Roadmap

* [ ] Define core body scan data model
* [ ] Implement data processing pipeline
* [ ] Add visualization capabilities
* [ ] Improve test coverage
* [ ] Publish package to PyPI

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with improvements, bug fixes, or feature proposals.

## License

Specify the project license here.
