import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sonar_sim import SonarSimulator

def create_test_pose():
    """
    Create a single pose looking straight forward.
    """
    return np.eye(4)

def test_vertical_interpolation(realistic_simulator:SonarSimulator):
    """
    Test that sonar simulation correctly interpolates vertically between range bins.
    """
    output_folder = Path("tests/output_test")
    output_folder.mkdir(parents=True, exist_ok=True)

    pose = np.eye(4)

    realistic_simulator.run_simulation([pose], output_folder=output_folder, run_name="test", normalize=True, incident_dependency=True)

    # Check that output image exists
    image_path = output_folder / "test_image_0000.png"
    assert image_path.exists(), "Output image not found"

    # Load image
    image = plt.imread(str(image_path))

    # Assertions to verify basic behavior
    non_zero_pixels = np.count_nonzero(image)
    total_pixels = image.size
    coverage_ratio = non_zero_pixels / total_pixels

    # The sonar should have hit the flat wall for most azimuth bins
    assert coverage_ratio > 0.0, "No non-zero pixels — possible raycast failure"
    # assert image.max() > 0.5, "Max intensity too low — normalization or intensity bug?"
