import yaml
import os
from dataclasses import dataclass

@dataclass
class RunConfig:
    name: str
    trajectory_file: str
    sonar_config_file: str
    scene_file: str
    output_folder: str

    @classmethod
    def from_yaml(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[RunConfig]  Error: Run config YAML file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        required_keys = ['name', 'trajectory_file', 'sonar_config_file', 'scene_file', 'output_folder']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"[RunConfig]  Error: Missing required key '{key}' in run config YAML.")

        return cls(
            name=data['name'],
            trajectory_file=data['trajectory_file'],
            sonar_config_file=data['sonar_config_file'],
            scene_file=data['scene_file'],
            output_folder=data['output_folder']
        )
    
@dataclass
class SonarConfig:
    v_fov: float
    h_fov: float
    max_range: float
    azimuth_bins: int
    range_bins: int
    elevation_bins: int

    @classmethod
    def from_yaml(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[SonarConfig]  Error: Sonar config YAML file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        required_keys = ['v_fov', 'h_fov', 'max_range', 'azimuth_bins', 'range_bins', 'elevation_bins']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"[SonarConfig]  Error: Missing required key '{key}' in sonar config YAML.")

        return cls(
            v_fov=float(data['v_fov']),
            h_fov=float(data['h_fov']),
            max_range=float(data['max_range']),
            azimuth_bins=int(data['azimuth_bins']),
            range_bins=int(data['range_bins']),
            elevation_bins=int(data['elevation_bins']),
        )

    def write_config(self, path: str):
        azimuth_resolution = self.h_fov / self.azimuth_bins
        range_resolution = self.max_range / self.range_bins
        
        # Prepare dictionary to write
        config_dict = {
            'v_fov': self.v_fov,
            'h_fov': self.h_fov,
            'max_range': self.max_range,
            'azimuth_bins': self.azimuth_bins,
            'range_bins': self.range_bins,
            'elevation_bins': self.elevation_bins,
            'azimuth_resolution': azimuth_resolution,
            'range_resolution': range_resolution
        }
        
        # Write to YAML file
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)