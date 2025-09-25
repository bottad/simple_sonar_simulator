import os
import numpy as np
from PIL import Image
import csv
import trimesh
# from trimesh.ray.ray_pyembree import RayMeshIntersector # pip install pyembree
from trimesh.ray.ray_triangle import RayMeshIntersector
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
        if mesh.is_empty or len(mesh.faces) == 0:
            print("[SonarSimulator]\tWarning: Empty mesh provided, raycasting disabled.")
            self.intersector = None
        else:
            self.intersector = RayMeshIntersector(mesh)

    def run_simulation(self, poses, output_folder, run_name="sonar", normalize=False, smoothing_sigma=None):
        os.makedirs(output_folder, exist_ok=True)

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

        Returns an image with shape (range_bins, azimuth_bins),
        where height = range and width = azimuth.
        """
        config = self.config
        sonar_image = np.zeros((config.range_bins, config.azimuth_bins), dtype=np.float32)

        if self.intersector is None:
            return sonar_image  # No mesh to intersect

        # Angular sampling
        azimuth_angles = np.linspace(
            -config.h_fov / 2, config.h_fov / 2, config.azimuth_bins, endpoint=False
        )
        elevation_angles = np.linspace(
            -config.v_fov / 2, config.v_fov / 2, config.elevation_bins, endpoint=False
        )

        azimuth_angles = np.radians(azimuth_angles)
        elevation_angles = np.radians(elevation_angles)

        # Sensor pose
        position = pose_matrix[:3, 3]
        rotation = pose_matrix[:3, :3]

        rays = []
        ray_indices = []

        for az_idx, az in enumerate(azimuth_angles):
            for el_idx, el in enumerate(elevation_angles):
                # Spherical to Cartesian (sensor local frame)
                direction_local = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el)
                ])
                direction_world = rotation @ direction_local
                rays.append(direction_world)
                ray_indices.append((az_idx, el_idx))

        # Cast rays
        rays = np.stack(rays, axis=0)
        origins = np.tile(position, (rays.shape[0], 1))

        # Intersect rays with mesh
        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=origins,
            ray_directions=rays,
            multiple_hits=False
        )

        if len(index_ray) == 0:
            return sonar_image  # No hits

        # Compute distances
        hit_vectors = locations - origins[index_ray]
        distances = np.linalg.norm(hit_vectors, axis=1)

        # Get surface normals from mesh
        normals = self.intersector.mesh.face_normals[index_tri]

        # Normalize ray directions
        ray_directions = rays[index_ray]
        ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

        # Incident angle (cosine)
        cos_incident = np.abs(np.einsum('ij,ij->i', ray_directions, normals))
        cos_incident = np.clip(cos_incident, 0.0, 1.0)

        for i, ray_i in enumerate(index_ray):
            az_idx, el_idx = ray_indices[ray_i]
            dist = distances[i]

            if dist <= config.max_range:
                # Map distance to range bin
                range_bin = int((dist / config.max_range) * config.range_bins)
                if range_bin >= config.range_bins:
                    continue

                # Intensity based on angle and vertical sampling
                base_intensity = cos_incident[i] / config.elevation_bins

                # Spread based on incident angle
                spread_bins = int((1 - cos_incident[i]) * dist / self.config.max_range * self.max_spread_bins)

                for offset in range(-spread_bins, spread_bins + 1):
                    target_bin = range_bin + offset
                    if 0 <= target_bin < config.range_bins:
                        # Gaussian falloff for spreading
                        falloff = np.exp(-0.5 * (offset / (spread_bins + 1e-5))**2)
                        sonar_image[target_bin, az_idx] += base_intensity * falloff

        return sonar_image
