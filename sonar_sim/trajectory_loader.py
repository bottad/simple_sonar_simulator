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

    start_pos = np.array(data['start_pose']['position'])
    start_rot = R.from_euler('xyz', data['start_pose']['rotation_euler'], degrees=True)

    end_pos = np.array(data['end_pose']['position'])
    end_rot = R.from_euler('xyz', data['end_pose']['rotation_euler'], degrees=True)

    total_time = data['total_time']
    timesteps = data['timesteps']

    dt = total_time / (timesteps - 1)

    poses = []

    # scipy Rotation.slerp requires a special call, so we build a slerp object
    slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))

    for t in range(timesteps):
        alpha = t / (timesteps - 1)

        # Interpolate position linearly
        pos = (1 - alpha) * start_pos + alpha * end_pos

        # Interpolate rotation via slerp
        rot = slerp(alpha)

        # Compose homogeneous transformation matrix
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = rot.as_matrix()
        pose_mat[:3, 3] = pos

        poses.append(pose_mat)

    return poses, dt
