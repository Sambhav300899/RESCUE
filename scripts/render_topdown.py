"""
Render a GLB reconstruction as a top-down orthographic image.
Only the point cloud geometry is rendered (camera frustum meshes are skipped).

Usage:
    python scripts/render_topdown.py [--input PATH] [--output PATH] [--res INT] [--up-axis INT]
"""

import argparse
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image

from rescue.utils import look_at


def render_topdown(
    recon_path,
    output_path,
    resolution=1024,
    bg_color=(255, 255, 255, 255),
    ambient_light=200,
):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    scene = trimesh.load(recon_path)

    # Pull out only PointCloud geometries (geometry_0); skip camera frustum meshes
    point_clouds = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.PointCloud)}
    print(f"Found {len(scene.geometry)} geometries, {len(point_clouds)} point cloud(s)")

    if not point_clouds:
        raise ValueError("No PointCloud geometry found in GLB — check with inspect_glb.py")

    # Merge all point clouds (usually just one: geometry_0)
    all_points = np.concatenate([pc.vertices for pc in point_clouds.values()], axis=0)
    all_colors = np.concatenate([pc.colors   for pc in point_clouds.values()], axis=0)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    extents = maxs - mins
    center  = (mins + maxs) / 2.0

    print(f"Point cloud: {len(all_points):,} points")
    print(f"Extents (x, y, z): {extents}")
    print(f"Centroid:           {center}")

    # World is ENU (East-North-Up): X=East, Y=North, Z=Up
    # "Above" the scene = +Z direction; screen-up when looking down -Z = +Y (North)
    height = extents[2] * 2 + 1.0
    eye = center.copy()
    eye[2] += height          # move in +Z = real-world up

    screen_up = np.array([0.0, 1.0, 0.0])   # +Y (North) appears as up in the image

    print(f"Eye: {eye}  (above scene in +Z, height={height:.3f})")
    print(f"Horizontal extents — X: {extents[0]:.3f}, Y: {extents[1]:.3f}")

    camera_pose = look_at(eye, center, screen_up)
    mag = max(extents[0], extents[1]) / 2.0   # XY (East-North) footprint

    ambient = np.ones(4, dtype="uint8") * ambient_light
    render_scene = pyrender.Scene(bg_color=list(bg_color), ambient_light=ambient)

    # pyrender.Mesh.from_points renders a point cloud with per-vertex colors
    pc_mesh = pyrender.Mesh.from_points(all_points, colors=all_colors)
    render_scene.add(pc_mesh)

    camera = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=0.01, zfar=height * 4)
    render_scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = r.render(render_scene)
    r.delete()

    img = Image.fromarray(color)
    img.save(output_path)
    print(f"Saved → {output_path}")
    return color


def main():
    parser = argparse.ArgumentParser(description="Top-down render of a GLB reconstruction")
    parser.add_argument("--input",   default="../generated/reconstruction.glb", help="Input GLB path")
    parser.add_argument("--output",  default="../generated/topdown_render.png",  help="Output PNG path")
    parser.add_argument("--res",     type=int, default=1024, help="Output resolution in pixels (square)")
    args = parser.parse_args()

    render_topdown(
        recon_path=args.input,
        output_path=args.output,
        resolution=args.res,
    )


if __name__ == "__main__":
    main()
