# Simple Sonar Simulator

This is a **simplified 3D sonar simulator** that casts rays into a 3D scene (using a triangle mesh) to generate synthetic sonar images. It is intended for testing and prototyping sonar-based perception pipelines.

---

## Features

- Simulates sonar images from 3D scenes (`.ply` files)
- Configurable vertical/horizontal field of view, resolution, range
- Trajectory-based simulation from start to end pose
- Outputs both `.png` images and a CSV log with poses
- Uses `trimesh` for geometry and ray intersection
- Extremely simplified — [**see limitations**](#simplifications-and-limitations)

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
python run_simulation.py run_config.yaml
```

The program will:

* Load configuration
* Load the scene mesh
* Interpolate poses along the trajectory
* Simulate sonar returns
* Save images + pose log

---

## Required Input Files

All input files are referenced in a central `run_config.yaml` file (located in `/config/`).

### 1. `run_config.yaml`
Specifies paths to:
```yaml
trajectory_file: config/trajectory.yaml
sonar_config_file: config/sonar_config.yaml
scene_file: scenes/example_scene.ply
output_folder: output/
```

### 2. `trajectory.yaml`

Defines start & end poses (Euler angles + position), total time, and number of steps.

```yaml
start_pose:
  position: [0, 0, 0]
  rotation_euler: [0, 0, 0]  # Degrees

end_pose:
  position: [5, 0, 0]
  rotation_euler: [0, 0, 0]

total_time: 10.0
timesteps: 100
```

### 3. `sonar_config.yaml`

Defines sonar parameters:

```yaml
v_fov: 30            # Vertical field of view in degrees
h_fov: 90            # Horizontal field of view in degrees
max_range: 10.0      # Maximum range in meters
azimuth_bins: 512
range_bins: 256
elevation_bins: 16
```

### 4. `scene_file.ply`

3D mesh of the environment (e.g., walls, terrain, etc.).

---

## Output

The simulator will generate:

* A series of sonar images in `.png` format:

  ```
  sonar_image_0000.png
  sonar_image_0001.png
  ...
  ```

* A `poses.csv` file containing:

  * Filename
  * Flattened 4x4 pose matrix for each frame

---

## Project Structure

```
simple_sonar_simulator/
│
├── config/
│   ├── run_config.yaml
│   ├── trajectory.yaml
│   └── sonar_config.yaml
│
├── scenes/
│   └── example_scene.ply
│
├── output/
│
├── sonar_sim/
│   ├── sonar_simulator.py
│   ├── data_classes.py
│   ├── trajectory_loader.py
│   ├── config_loader.py
│   └── scene_loader.py
│
├── run_simulation.py
└── requirements.txt
```
---

## Simplifications and Limitations

> **This sonar simulator is intentionally simplified.**

It is designed primarily for **educational use, algorithm prototyping**, or **synthetic data generation** under idealized assumptions. The following sonar effects are **not simulated**:

* ❌ **Surface reflectance** or material interaction
* ❌ **Acoustic scattering models**
* ❌ **Echo strength attenuation**
* ❌ **Surface normal interaction**
* ❌ **Beam patterns**, sound velocity, or environmental effects
* ❌ **Multiple reflections**, reverberation, or noise

### Simulated:

* ✔️ Ray-based first-return logic
* ✔️ Configurable FOV, resolution, and max range
* ✔️ Simple geometric intersection
* ✔️ Consistent sonar image generation
