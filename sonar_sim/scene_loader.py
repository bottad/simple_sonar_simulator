import os
import trimesh

def load_scene(path: str) -> trimesh.Trimesh:
    """
    Load a 3D scene from a mesh file (.ply, .obj, etc).

    Parameters:
        path (str): Path to the scene file.

    Returns:
        trimesh.Trimesh: Loaded and processed mesh.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[scene_loader]  Error: Scene file not found: {path}")

    # Load mesh or scene
    loaded = trimesh.load(path)

    # Convert to Trimesh if scene is returned
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            print(f"[scene_loader]  Warning: Scene loaded from '{path}' contains no geometry. Proceeding with empty environment.")
        combined = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        mesh = combined
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"[scene_loader]  Error: Unsupported mesh type: {type(loaded)}")

    # Optional: clean the mesh
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()

    if not mesh.is_watertight:
        print("[scene_loader]  Warning: Mesh is not watertight. May affect raycasting.")

    return mesh
