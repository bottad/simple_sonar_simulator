import numpy as np
import pytest
import trimesh
from sonar_sim import SonarSimulator, SonarConfig

def create_test_plane():
    # Plane 10 m ahead, wide enough to cover the FOV at max range
    x = 10.0
    y_extent = 25.0  # half-width
    z_extent = 2.0   # half-height

    vertices = np.array([
        [x, -y_extent, -z_extent],
        [x,  y_extent, -z_extent],
        [x,  y_extent,  z_extent],
        [x, -y_extent,  z_extent]
    ])
    faces = np.array([
        [0,1,2],
        [0,2,3]
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

import numpy as np
import trimesh

def create_test_split():
    y_extent = 25.0  # half-width on y-axis

    # Lower plane at x=10, covers z from -2 to -1 (leaving gap at z = -1 to 1)
    x_lower = 10.0
    z_lower_min, z_lower_max = -3.0, -1.0

    vertices_lower = np.array([
        [x_lower, -y_extent, z_lower_min],
        [x_lower,  y_extent, z_lower_min],
        [x_lower,  y_extent, z_lower_max],
        [x_lower, -y_extent, z_lower_max]
    ])

    faces_lower = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # Upper plane at x=15, covers z from 1 to 2
    x_upper = 15.0
    z_upper_min, z_upper_max = 1.0, 3.0

    vertices_upper = np.array([
        [x_upper, -y_extent, z_upper_min],
        [x_upper,  y_extent, z_upper_min],
        [x_upper,  y_extent, z_upper_max],
        [x_upper, -y_extent, z_upper_max]
    ])

    faces_upper = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # Combine vertices and faces, adjusting indices for upper plane faces
    vertices = np.vstack([vertices_lower, vertices_upper])
    faces_upper_adjusted = faces_upper + len(vertices_lower)
    faces = np.vstack([faces_lower, faces_upper_adjusted])

    # Create combined mesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def create_test_slopes():
    y_extent = 25.0  # half-width (y direction)
    z_gap = 1.0      # gap size in vertical

    # Define slopes (tangents of angles)
    slope1 = 0.2  # smaller positive slope for lower plane
    slope2 = 0.5  # larger positive slope for upper plane

    # Define z ranges for the two planes (with gap between -1 and 1)
    z1_min, z1_max = -3.0, -z_gap
    z2_min, z2_max = z_gap, 3.0

    # Generate vertices for lower sloped plane (z < 0)
    # x changes with z according to slope1, y spans full extent
    lower_vertices = np.array([
        [10.0 + slope1 * z1_min, -y_extent, z1_min],  # bottom left
        [10.0 + slope1 * z1_min,  y_extent, z1_min],  # bottom right
        [10.0 + slope1 * z1_max,  y_extent, z1_max],  # top right
        [10.0 + slope1 * z1_max, -y_extent, z1_max],  # top left
    ])

    # Generate vertices for upper sloped plane (z > 0)
    # x changes with z according to slope2, y spans full extent
    upper_vertices = np.array([
        [15.0 + slope2 * z2_min, -y_extent, z2_min],  # bottom left
        [15.0 + slope2 * z2_min,  y_extent, z2_min],  # bottom right
        [15.0 + slope2 * z2_max,  y_extent, z2_max],  # top right
        [15.0 + slope2 * z2_max, -y_extent, z2_max],  # top left
    ])

    # Combine vertices
    vertices = np.vstack([lower_vertices, upper_vertices])

    # Define faces for two quads (split into triangles)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # lower plane
        [4, 5, 6], [4, 6, 7],  # upper plane
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def create_horizontal_plane():
    x_extent_min, x_extent_max = 0.0, 40.0  # 4 meters wide in x
    y_extent = 25.0  # half-width
    z = -1.0         # fixed height

    vertices = np.array([
        [x_extent_min, -y_extent, z],  # bottom-left corner
        [x_extent_max, -y_extent, z],  # bottom-right corner
        [x_extent_max, y_extent, z],  # top-right corner
        [x_extent_min, y_extent, z],  # top-left corner
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)

@pytest.fixture(scope="module") 
def realistic_simulator():
    config = SonarConfig(
        azimuth_bins=512,
        elevation_bins=256,
        range_bins=512,
        max_range=20.0,
        h_fov=65.0,
        v_fov=20.0
    )
    mesh = create_horizontal_plane()
    sim = SonarSimulator(config=config, mesh=mesh)
    return sim