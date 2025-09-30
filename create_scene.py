# Define all vertices
vertices = {
    # Base corners (for floor)
    "Base1": (-60, -60, -1),
    "Base2": (-60,  60, -1),
    "Base3": ( 60,  60, -1),
    "Base4": ( 60, -60, -1),
    "TopBase1": (-60, -60, 0),
    "TopBase2": (-60,  60, 0),
    "TopBase3": ( 60,  60, 0),
    "TopBase4": ( 60, -60, 0),

    # Ground level of structure (z=0)
    "A": (-10, -30, 0),
    "B": (-10,  30, 0),
    "C": ( 40,  30, 0),
    "D": ( 40, -30, 0),
    "P1": (20, -30, 0),
    "P2": (20, 0, 0),
    "P3": (40, 0, 0),

    # Upper levels
    "E": (-10, -30, 20),
    "F": (-10,  30, 20),
    "G": (  0,  30, 20),
    "H": (  0, -30, 20),

    "I": (10, 0, 10),
    "J": (10, 30, 10),
    "K": (20, 30, 10),
    "L": (20, 0, 10),
}

# List of vertices and name lookup
vertex_names = list(vertices.keys())
vertex_list = list(vertices.values())

# Helper to convert names to indices
def face(*pts):
    return tuple(vertex_names.index(p) for p in pts)

# Define faces
faces = []

# Base (bottom and top)
faces += [
    face("Base1", "Base2", "Base3"), face("Base1", "Base3", "Base4"),
    face("TopBase1", "TopBase4", "TopBase3"), face("TopBase1", "TopBase3", "TopBase2"),

    # Base side walls
    face("Base1", "TopBase1", "TopBase2"), face("Base1", "TopBase2", "Base2"),
    face("Base2", "TopBase2", "TopBase3"), face("Base2", "TopBase3", "Base3"),
    face("Base3", "TopBase3", "TopBase4"), face("Base3", "TopBase4", "Base4"),
    face("Base4", "TopBase4", "TopBase1"), face("Base4", "TopBase1", "Base1"),
]

# Structure vertical walls
faces += [
    face("A", "B", "F"), face("A", "F", "E"),             # Wall A-B-F-E
    face("A", "E", "H"), face("A", "H", "P1"),            # Wall A-E-H-P1
    face("B", "F", "G"), face("B", "G", "J"),             # Wall B-F-G-J
    face("J", "K", "C"), face("J", "C", "B"),             # Wall J-K-C-B
    face("I", "L", "P3"), face("I", "P3", "P2"),          # Wall I-L-P3-P2
]

# Sloped surfaces
faces += [
    face("H", "P1", "P2"), face("H", "P2", "G"),          # Slope H-P1-P2-G
    face("P3", "C", "K"), face("P3", "K", "L"),           # Slope P3-C-K-L
    face("J", "G", "I"),                                  # Slope J-G-I
]

# Horizontal top faces
faces += [
    face("E", "F", "G"), face("E", "G", "H"),             # Top face at z=20
    face("I", "J", "K"), face("I", "K", "L"),             # Top face at z=10
]

# Save to PLY (ASCII)
def write_ply_ascii(vertices, faces, filename):
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face_ in faces:
            f.write(f"3 {' '.join(map(str, face_))}\n")

# Run and generate the file
write_ply_ascii(vertex_list, faces, "output/custom_structure.ply")
print("✅ PLY file 'custom_structure.ply' has been written.")
