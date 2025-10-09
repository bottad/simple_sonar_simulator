# Simple Sonar Simulator

This is a **simplified 3D sonar simulator** that casts rays into a 3D mesh scene to generate synthetic sonar images. It is intended for testing and prototyping sonar-based perception pipelines and generating synthetic data under controlled conditions.

---

## Features

- Simulates sonar images from 3D triangle meshes (`.ply`)
- Configurable sonar parameters (FOV, resolution, range)
- Supports keyframe-based trajectory definition and interpolation
- Per-ray surface interaction using incident angle
- Range- and angle-dependent return spreading across bins
- Optional Gaussian smoothing and image normalization
- Outputs PNG images and a CSV log of poses
- Fast mesh intersection using `trimesh`
- Simple and efficient — [**see limitations**](#simplifications-and-limitations)

---

## Contents

- [Installation](#installation)
- [Running the Simulation](#running-the-simulation)
- [Required Input Files](#required-input-files)
  - [`run_config.yaml`](#1-run_configyaml)
  - [`trajectory.yaml`](#2-trajectoryyaml)
  - [`sonar_config.yaml`](#3-sonar_configyaml)
  - [`scene_file.ply`](#4-scene_fileply)
- [Output](#output)
- [Project Structure](#project-structure)
- [⚠️ Simplifications & Limitations](#simplifications-and-limitations)

---

## Installation

Recommended: use a Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python run_simulation.py example_run_config.yaml
```

The simulator will:

* Load mesh and configuration files
* Interpolate poses along the trajectory
* Simulate sonar returns using ray intersection
* Save sonar images and log pose data

> ⚠️ You **do not need to specify a full path** to the config file — the simulator assumes it is located in the `config/` folder.
> Ensure that `run_config.yaml` (and all referenced paths inside it) are correctly placed in `config/`.

## Required Input Files

All input files are referenced in a central `run_config.yaml` file (located in `/config/`). This file also contains a few configuration options for the simulation itself. A full set of example files are provided within the repository. 

### `run_config.yaml`

```yaml
trajectory_file: config/example_trajectory.yaml
sonar_config_file: config/example_sonar_config.yaml
scene_file: scenes/example.ply
output_folder: output/
name: example            # Optional name prefix for images and logs
normalize: true          # Normalize sonar images to [0, 1] before saving
smoothing_sigma: 0.8     # Apply optional Gaussian smoothing (set to null to disable)
```

### `trajectory.yaml`

Defines a keyframe-based trajectory with linear interpolation between poses.

```yaml
poses:
  - position: [0, 1, 0]
    rotation_euler: [0, 0, 0]
  - position: [0, 2, -1]
    rotation_euler: [0, 0, 0]
  ...
  - position: [0, 3, -2]
    rotation_euler: [0, 20, 0]

total_time: 10.0      # Duration in seconds
timesteps: 40         # Number of interpolated steps
```

* `position`: XYZ in meters
* `rotation_euler`: rotation in degrees (XYZ order)
* Interpolation is linear in position and Euler angles

### `sonar_config.yaml`

Defines the sonar beam properties.

```yaml
v_fov: 30.0             # Vertical field of view in degrees
h_fov: 90.0             # Horizontal field of view in degrees
max_range: 10.0         # Maximum sensing range in meters

azimuth_bins: 512       # Horizontal resolution (image width)
range_bins: 256         # Range resolution (image height)
elevation_bins: 16      # Vertical sampling rays (not rendered)

```

### `scene_file.ply`

A triangle mesh (`.ply`) defining the 3D scene geometry.

* Must be a valid mesh (e.g., watertightness not required)
* Units should match sonar configuration (meters)

---

## Output

The simulator will generate:

* A series of sonar images in `.png` format:

  ```
  sonar_image_0000.png
  sonar_image_0001.png
  ...
  ```
  * Image dimensions: `(range_bins, azimuth_bins)`
  * Optional smoothing via Gaussian filter
  * Values normalized to `[0, 1]` if configured

* A `poses.csv` file containing:

  * Filename
  * Flattened 4x4 pose matrix for each frame

---

## Project Structure

```
simple_sonar_simulator/
│
├── config/
│   ├── example_run_config.yaml
│   ├── example_trajectory.yaml
│   └── example_sonar_config.yaml
│
├── scenes/
│   └── example.ply
│
├── output/
│
├── sonar_sim/
│   ├── __init__.py
│   ├── sonar_simulator.py
│   ├── data_classes.py
│   ├── trajectory_loader.py
│   ├── config_loader.py
│   └── scene_loader.py
│
├── tests/
│   ├── __init__.py
│   └── test_sonar_interpolation.py
│
├── run_simulation.py
└── requirements.txt
```
---

## Simplifications and Limitations

> **This sonar simulator is intentionally simplified.**

It is designed primarily for **educational use, algorithm prototyping**, or **synthetic data generation** under idealized assumptions. The following sonar effects are:

### Not Simulated

* ❌ Acoustic reflectance or material modeling
* ❌ Signal attenuation over distance
* ❌ Sound propagation, environmental effects, or noise
* ❌ Multi-path reflections, reverberation
* ❌ Beam patterns, sound speed gradients

### Simulated

* ✅ First-return ray intersection
* ✅ Brightness scaling based on incident angle
* ✅ Return spreading based on incident angle and range
* ✅ Keyframe-based trajectory interpolation
* ✅ Image output with optional smoothing and normalization
