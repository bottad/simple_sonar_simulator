import numpy as np
import pytest
from sonar_sim import SonarSimulator

def test_generate_column_rays(realistic_simulator:SonarSimulator):
    az_idx = 10  # arbitrary azimuth bin
    pose_matrix = np.eye(4)

    origins, directions, ray_indices = realistic_simulator._generate_rays(az_idx, pose_matrix)

    assert origins.shape == (realistic_simulator.config.elevation_bins, 3)
    assert directions.shape == (realistic_simulator.config.elevation_bins, 3)
    assert len(ray_indices) == realistic_simulator.config.elevation_bins

    # Directions should be unit vectors approximately
    norms = np.linalg.norm(directions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_intersect_rays(realistic_simulator: SonarSimulator):
    az_idx = realistic_simulator.config.azimuth_bins // 2  # use integer division for valid index
    pose_matrix = np.eye(4)

    origins, directions, _ = realistic_simulator._generate_rays(az_idx, pose_matrix)
    distances, index_ray, locations, index_tri = realistic_simulator._intersect_rays(origins, directions)

    # Check that some rays did intersect
    assert len(index_ray) > 0, f"No intersections found — unexpected for this mesh at azimuth index {az_idx}"

    # Check that all returned distances are within max_range
    assert np.all(distances <= realistic_simulator.config.max_range), "Some distances exceed max range"

    # Ensure distances shape matches index_ray shape
    assert distances.shape == index_ray.shape, "Mismatch between distances and ray indices"

    # Ensure locations and face indices also match
    assert locations.shape[0] == distances.shape[0], "Mismatch between locations and distances"
    assert index_tri.shape[0] == distances.shape[0], "Mismatch between triangle indices and distances"

    # Check that no ray index is repeated
    unique_indices, counts = np.unique(index_ray, return_counts=True)
    assert np.all(counts == 1), "Some ray indices have multiple intersections — expected only one per ray"
