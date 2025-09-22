import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def load_trajectory(yaml_path: str):
    """
    Load trajectory from a YAML file and generate interpolated poses.

    Parameters:
        yaml_path (str): Path to the YAML trajectory file.

    Returns:
        poses (list of np.ndarray): List of 4x4 homogeneous matrices.
        dt (float): Time interval between poses.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    keyframe_positions = []
    keyframe_rotations = []

    for pose in data['poses']:
        keyframe_positions.append(np.array(pose['position']))
        rot = R.from_euler('xyz', pose['rotation_euler'], degrees=True)
        keyframe_rotations.append(rot)

    total_time = data['total_time']
    timesteps = data['timesteps']
    dt = total_time / (timesteps - 1)

    # Total number of keyframes
    num_keyframes = len(keyframe_positions)
    segment_count = num_keyframes - 1

    # Time allocation: distribute timesteps evenly across segments
    timesteps_per_segment = np.linspace(0, timesteps - 1, segment_count + 1, dtype=int)

    poses = []

    for i in range(segment_count):
        start_idx = timesteps_per_segment[i]
        end_idx = timesteps_per_segment[i + 1]
        steps = end_idx - start_idx + 1

        if steps <= 1:
            continue  # Nothing to interpolate

        # Setup Slerp between current segment's rotations
        slerp_times = [0, 1]
        slerp_rots = R.concatenate([keyframe_rotations[i], keyframe_rotations[i + 1]])
        slerp = Slerp(slerp_times, slerp_rots)

        for j in range(steps):
            alpha = j / (steps - 1)

            # Interpolate position
            pos = (1 - alpha) * keyframe_positions[i] + alpha * keyframe_positions[i + 1]

            # Interpolate rotation
            rot = slerp(alpha)

            # Compose transformation matrix
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = rot.as_matrix()
            pose_mat[:3, 3] = pos

            # Avoid duplicate poses between segments
            if i > 0 and j == 0:
                continue

            poses.append(pose_mat)

    return poses, dt
