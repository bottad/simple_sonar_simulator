import os
import argparse

from sonar_sim.data_classes import *
from sonar_sim.trajectory_loader import load_trajectory
from sonar_sim.scene_loader import load_scene
from sonar_sim.sonar_simulator import SonarSimulator

config_folder = "config"

def main(run_config_filename: str):
    # Construct full path to run config
    run_config_path = os.path.join(config_folder, run_config_filename)

    # Load run config
    run_config = RunConfig.from_yaml(run_config_path)
    print(f"Loaded run config from {run_config_path}")
    
    # Load sonar config
    sonar_config = SonarConfig.from_yaml(run_config.sonar_config_file)
    print(f"Loaded sonar config from {run_config.sonar_config_file}")
    
    # Load trajectory
    poses, _ = load_trajectory(run_config.trajectory_file)
    print(f"Loaded trajectory from {run_config.trajectory_file}")
    
    # Load scene
    scene = load_scene(run_config.scene_file)
    print(f"Loaded scene from {run_config.scene_file}")

    # Initialize simulator
    simulator = SonarSimulator(sonar_config, scene)

    # Run the simulation
    simulator.run_simulation(poses, run_config.output_folder, run_config.name)
    print(f"Simulation completed. Results saved to: {run_config.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sonar simulation based on a run configuration file."
    )
    parser.add_argument(
        "run_config",
        type=str,
        help="Name of the run config YAML file located in the 'config/' folder"
    )

    args = parser.parse_args()
    main(args.run_config)
