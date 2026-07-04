# BodyScan

BodyScan is a Python package designed to provide functionality for body analysis and scanning workflows by extracting kinectic metrices from a moving human object. This repository contains the core package, project configuration, and supporting code for development and deployment. The package is build around the 'KineticBody' class, which computes 

## Overview

The goal of BodyScan is to provide a modular and extensible foundation for processing, analyzing, and managing body scan data.

## Features

<<<<<<< HEAD
- Pose estimation for `.mp4`, `.mov`, `.png`, `.jpg`, and `.jpeg` inputs
- `KineticBody` model export as JSON
- Improved performance compared to the baseline MediaPipe Pose Landmarker by applying lag reduction and Savitzky-Golay filtering for video inference (see below).
- Video visualization from a saved body model
- Docker support for running pose estimation in a reproducible environment

## Demo

The front-view squat example below compares the KineticBody visualization with a
plain MediaPipe visualization.

<table>
  <tr>
    <th>KineticBody</th>
    <th>Plain MediaPipe</th>
  </tr>
  <tr>
    <td>
      <video src="docs/videos/squat_front_view_shortened.mov" controls width="360"></video>
      <br>
      <a href="docs/videos/squat_front_view_shortened.mov">Open KineticBody video</a>
    </td>
    <td>
      <video src="docs/videos/squat_front_view_shortened_mp_only.mov" controls width="360"></video>
      <br>
      <a href="docs/videos/squat_front_view_shortened_mp_only.mov">Open plain MediaPipe video</a>
    </td>
  </tr>
</table>

## Project Structure

```text
KineticBody/
├── docs/
│   └── videos/
├── docker/
│   ├── pose_estimation.dockerfile
│   └── requirements.txt
├── scripts/
│   ├── build_container.bash
│   ├── run_pose_estimation.py
│   ├── run_pose_estimation_container.bash
│   └── visualize_body.py
├── src/
│   └── kineticbody/
│       ├── config/
│       ├── kinetics/
│       ├── model/
│       ├── utils/
│       ├── visualization/
│       └── workflows/
├── pyproject.toml
├── requirements.txt
└── README.md
```
=======
* Python package structure using modern packaging standards
* Easy installation via `pip`
* Extensible architecture for future body analysis capabilities
* Development-friendly project layout
>>>>>>> parent of d61346e (added documentation videos // worked on README)

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
