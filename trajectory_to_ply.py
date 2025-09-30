import yaml
import argparse

def read_trajectory_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    positions = [tuple(pose["position"]) for pose in data["poses"]]
    return positions

def write_trajectory_ply(positions, ply_file):
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(positions) - 1}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # Write vertex list
        for pos in positions:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Write edges to connect the points in sequence
        for i in range(len(positions) - 1):
            f.write(f"{i} {i + 1}\n")

    print(f"✅ Wrote trajectory to: {ply_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert trajectory YAML to PLY line file.")
    parser.add_argument("-i", "--input", help="Path to the trajectory YAML file")
    parser.add_argument("-o", "--output", default="output/trajectory_path.ply", help="Output PLY file path")
    args = parser.parse_args()

    positions = read_trajectory_yaml(args.input)
    write_trajectory_ply(positions, args.output)

if __name__ == "__main__":
    main()
