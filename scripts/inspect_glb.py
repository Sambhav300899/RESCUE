"""
Inspect the contents of a GLB file — geometries, types, sizes, scene graph.

Usage:
    python scripts/inspect_glb.py [--input PATH]
"""

import argparse
import trimesh
import numpy as np


def inspect(path):
    scene = trimesh.load(path)

    print(f"\n=== {path} ===")
    print(f"Total geometries: {len(scene.geometry)}\n")

    print(f"{'Name':<45} {'Type':<15} {'Vertices':>10} {'Faces':>10} {'Notes'}")
    print("-" * 100)

    for name, geom in scene.geometry.items():
        n_verts = len(geom.vertices)
        n_faces = len(geom.faces) if hasattr(geom, "faces") and geom.faces is not None else 0
        kind = type(geom).__name__

        notes = []
        if isinstance(geom, trimesh.PointCloud):
            notes.append("POINT CLOUD")
        elif n_faces == 0:
            notes.append("no faces")
        if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
            vc = geom.visual.vertex_colors
            if vc is not None and len(vc):
                notes.append("has vertex colors")

        print(f"{name:<45} {kind:<15} {n_verts:>10} {n_faces:>10}  {', '.join(notes)}")

    print("\n=== Scene graph nodes ===")
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node]
        translation = T[:3, 3]
        print(f"  {node:<45}  geom={geom_name}  translation={np.round(translation, 3)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect contents of a GLB file")
    parser.add_argument("--input", default="../generated/reconstruction.glb", help="GLB file to inspect")
    args = parser.parse_args()
    inspect(args.input)


if __name__ == "__main__":
    main()
