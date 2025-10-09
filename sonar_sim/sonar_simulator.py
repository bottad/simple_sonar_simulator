import os
import numpy as np
from PIL import Image
import csv
import trimesh
from typing import Tuple, List
from trimesh.ray.ray_pyembree import RayMeshIntersector # pip install pyembree
# from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.ndimage import gaussian_filter

from .data_classes import SonarConfig

def save_image_as_png(image: np.ndarray, output_path: str, normalize: bool):
    """
    Save a sonar image to disk as an 8-bit grayscale PNG.

    Parameters:
        image (np.ndarray): Input image as a 2D NumPy array.
        output_path (str): Path to save the output PNG file.
        normalize (bool): Whether to normalize the image to [0, 1] before saving.

    Behavior:
        - Image is clipped to [0, 1] before any further processing.
        - If `normalize` is True, the image is scaled by dividing by its maximum value,
          unless the maximum is 0, in which case normalization is skipped.
        - If `normalize` is False and image values are outside [0, 1], a warning is printed
          and normalization is still applied (unless max is 0).
        - The resulting image is converted to 8-bit grayscale and saved as a PNG.
    """
    # image = np.clip(image, 0.0, 1.0)
    max_val = image.max()

    if normalize and max_val > 0:
        image = image / max_val

    img_uint8 = (255 * image).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')  # 'L' = 8-bit grayscale
    img.save(output_path)

class SonarSimulator:
    def __init__(self, config: SonarConfig, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.config = config
        self.max_spread_bins = 7
        self.incident_dependency = False
        if mesh.is_empty or len(mesh.faces) == 0:
            print("[SonarSimulator]\tWarning: Empty mesh provided, raycasting disabled.")
            self.intersector = None
        else:
            self.intersector = RayMeshIntersector(mesh)

    def run_simulation(self, poses, output_folder, run_name="sonar", normalize=False, incident_dependency=False, smoothing_sigma=None):
        os.makedirs(output_folder, exist_ok=True)

        self.incident_dependency = incident_dependency

        # Print sonar configuration
        print(f"[SonarSimulator]  Info: Sonar configuration: {self.config}")
        config_path = os.path.join(output_folder, f"{run_name}_sonar_config.yaml")
        self.config.write_config(config_path)

        total = len(poses)
        csv_path = os.path.join(output_folder, f"{run_name}_data.csv")

        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ["image_filename"] + [f"pose_{i}" for i in range(16)]
            writer.writerow(header)

            for idx, pose in enumerate(poses):
                sonar_image = self.compute_sonar_image(pose)

                # apply smoothing
                if smoothing_sigma is not None:
                    sonar_image = gaussian_filter(sonar_image, sigma=smoothing_sigma)

                # Save image
                image_filename = f"{run_name}_image_{idx:04d}.png"
                image_path = os.path.join(output_folder, image_filename)
                save_image_as_png(sonar_image, image_path, normalize)

                # Write pose to CSV
                flat_pose = pose.flatten().tolist()
                writer.writerow([image_filename] + flat_pose)

                # Progress percentage
                percent = (idx + 1) / total * 100
                print(f"[SonarSimulator]  Info: Simulation progress: {percent:.1f}%   ", end='\r')

        print(f"\n[SonarSimulator]  Info: Saved {total} sonar images and poses to '{output_folder}'")

    def compute_sonar_image(self, pose_matrix: np.ndarray) -> np.ndarray:
        """
        Simulate a sonar image from the given 4x4 pose matrix.
        Returns an image with shape (range_bins, azimuth_bins).
        """
        config = self.config
        sonar_image = np.zeros((config.range_bins, config.azimuth_bins), dtype=np.float32)

        if self.intersector is None:
            return sonar_image

        for az_idx in range(config.azimuth_bins):
            origins, directions, ray_indices = self._generate_rays(az_idx, pose_matrix)
            distances, index_ray, locations, index_tri = self._intersect_rays(origins, directions)

            if len(index_ray) == 0:
                continue

            normals = self.intersector.mesh.face_normals[index_tri]
            ray_dirs = directions[index_ray]
            cos_incident = np.abs(np.einsum('ij,ij->i', ray_dirs, normals))
            cos_incident = np.clip(cos_incident, 0.0, 1.0)

            last_hit_ray_idx = None
            last_hit_range_bin = None
            last_intensity = None

            for i, ray_i in enumerate(index_ray):
                range_bin = int(round((distances[i] / config.max_range) * config.range_bins))
                if not (0 <= range_bin < config.range_bins):
                    continue

                # Compute intensity for this hit
                intensity = (cos_incident[i] if self.incident_dependency else 1.0)
                sonar_image[range_bin, az_idx] += intensity

                if last_hit_ray_idx is not None and ray_i == last_hit_ray_idx + 1:
                    start_bin = last_hit_range_bin
                    end_bin = range_bin

                    if end_bin - start_bin > 1:
                        # Interpolate between last and current intensity, excluding endpoints
                        interp_values = np.linspace(last_intensity, intensity, end_bin - start_bin + 1)[1:-1]
                        for j, val in enumerate(interp_values, start=start_bin + 1):
                            sonar_image[j, az_idx] += val

                last_hit_ray_idx = ray_i
                last_hit_range_bin = range_bin
                last_intensity = intensity

        return sonar_image

    def _generate_rays(self, az_idx: int, pose_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Generate all rays for a single azimuth index (i.e., a vertical column).
        Returns:
            - origins: (E, 3)
            - directions: (E, 3)
            - ray_indices: List of (az_idx, el_idx)
        """
        config = self.config
        az = np.radians(-config.h_fov / 2 + (az_idx + 0.5) * config.h_fov / config.azimuth_bins)
        el = np.radians(np.linspace(-config.v_fov / 2, config.v_fov / 2, config.elevation_bins, endpoint=False))

        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)

        directions_local = np.stack([x, y, z], axis=-1)  # (E, 3)
        rotation = pose_matrix[:3, :3]
        directions_world = directions_local @ rotation.T

        origin = pose_matrix[:3, 3]
        origins = np.tile(origin, (config.elevation_bins, 1))

        ray_indices = [(az_idx, el_idx) for el_idx in range(config.elevation_bins)]

        return origins, directions_world, ray_indices

    def _intersect_rays(self, origins: np.ndarray, directions: np.ndarray):
        """
        Intersect rays with the mesh and return information only for hits
        within the configured max_range.

        Args:
            origins (np.ndarray): Array of shape (N, 3) representing ray origins.
            directions (np.ndarray): Array of shape (N, 3) representing normalized ray directions.

        Returns:
            distances (np.ndarray): Distances from ray origins to intersection points (filtered by max_range).
            hit_ray_indices (np.ndarray): Indices of the rays that hit the mesh.
            hit_locations (np.ndarray): Intersection points on the mesh surface.
            hit_face_indices (np.ndarray): Indices of mesh faces that were hit.
        """
        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False
        )

        if len(index_ray) == 0:
            empty = np.empty(0)
            empty_int = np.empty(0, dtype=int)
            return empty, empty_int, empty.reshape((0, 3)), empty_int

        # Compute distances from ray origins to intersection points
        distances = np.linalg.norm(locations - origins[index_ray], axis=1)

        # Keep only hits within max range
        valid = distances <= self.config.max_range

        return distances[valid], index_ray[valid], locations[valid], index_tri[valid]

    def _spread_intensity(self, sonar_image, az_idx, range_bin, base_intensity, spread_bins):
        """
        Spread intensity across nearby range bins using normalized Gaussian falloff.
        """
        if spread_bins == 0:
            if 0 <= range_bin < self.config.range_bins:
                sonar_image[range_bin, az_idx] += base_intensity
            return

        offsets = np.arange(-spread_bins, spread_bins + 1)
        falloffs = np.exp(-0.5 * (offsets / (spread_bins + 1e-5))**2)
        falloffs /= np.sum(falloffs)  # Normalize

        for offset, falloff in zip(offsets, falloffs):
            target_bin = range_bin + offset
            if 0 <= target_bin < self.config.range_bins:
                sonar_image[target_bin, az_idx] += base_intensity * falloff
