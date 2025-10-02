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
        - If `normalize` is True, the image is scaled by dividing by its maximum value,
          unless the maximum is 0, in which case normalization is skipped.
        - If `normalize` is False and image values are outside [0, 1], a warning is printed
          and normalization is still applied (unless max is 0).
        - The resulting image is converted to 8-bit grayscale and saved as a PNG.
    """
    needs_normalization = not ((0.0 <= image).all() and (image <= 1.0).all())
    max_val = image.max()

    if normalize or needs_normalization:
        if max_val > 0:
            if needs_normalization:
                print(f"[save_image_as_png]  Warning: Image values outside [0, 1]. Forcing normalization.")
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

        origins, rays, ray_indices = self._generate_rays(pose_matrix)
        locations, index_ray, index_tri = self._intersect_rays(origins, rays)

        if len(index_ray) == 0:
            return sonar_image

        hit_vectors = locations - origins[index_ray]
        distances = np.linalg.norm(hit_vectors, axis=1)

        normals = self.intersector.mesh.face_normals[index_tri]
        ray_directions = rays[index_ray]
        ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

        cos_incident = np.abs(np.einsum('ij,ij->i', ray_directions, normals))
        cos_incident = np.clip(cos_incident, 0.0, 1.0)

        for i, ray_i in enumerate(index_ray):
            az_idx, el_idx = ray_indices[ray_i]
            dist = distances[i]

            if dist > config.max_range:
                continue

            range_bin = int((dist / config.max_range) * config.range_bins)
            if range_bin >= config.range_bins:
                continue

            if self.incident_dependency:
                base_intensity = cos_incident[i] / config.elevation_bins
            else:
                base_intensity = 1.0 / config.elevation_bins

            spread_bins = int((1 - cos_incident[i]) * dist / config.max_range * self.max_spread_bins)

            self._spread_intensity(sonar_image, az_idx, range_bin, base_intensity, spread_bins)

        return sonar_image

    def _generate_rays(self, pose_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Generate ray origins and directions in world space.
        Returns:
            - origins: (N, 3)
            - directions: (N, 3)
            - ray_indices: mapping to (azimuth_idx, elevation_idx)
        """
        config = self.config
        az = np.radians(np.linspace(-config.h_fov / 2, config.h_fov / 2, config.azimuth_bins, endpoint=False))
        el = np.radians(np.linspace(-config.v_fov / 2, config.v_fov / 2, config.elevation_bins, endpoint=False))
        az_grid, el_grid = np.meshgrid(az, el, indexing='ij')

        x = np.cos(el_grid) * np.cos(az_grid)
        y = np.cos(el_grid) * np.sin(az_grid)
        z = np.sin(el_grid)

        directions_local = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        rotation = pose_matrix[:3, :3]
        directions_world = directions_local @ rotation.T

        position = pose_matrix[:3, 3]
        origins = np.tile(position, (directions_world.shape[0], 1))

        # Generate az/el indices
        ray_indices = [(i, j) for i in range(config.azimuth_bins) for j in range(config.elevation_bins)]
        return origins, directions_world, ray_indices


    def _intersect_rays(self, origins: np.ndarray, directions: np.ndarray):
        """
        Intersect rays with the mesh.
        Returns:
            - hit_locations
            - hit_ray_indices
            - hit_face_indices
        """
        return self.intersector.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False
        )

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
