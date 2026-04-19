import cv2
import numpy as np
import torch
import trimesh
from transformers import AutoImageProcessor, SuperGlueForKeypointMatching
from safetensors.torch import save_file, load_file

class SuperGlueMatcher:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoImageProcessor.from_pretrained(
            "magic-leap-community/superglue_outdoor"
        )
        self.model = (
            SuperGlueForKeypointMatching.from_pretrained(
                "magic-leap-community/superglue_outdoor"
            )
            .to(self.device)
            .eval()
        )

    def match(self, img_1, img_2, threshold: float = 0.5):
        imgs = [img_1, img_2]
        inputs = self.processor(imgs, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        img_sizes = [[(img.height, img.width) for img in imgs]]
        processed_outputs = self.processor.post_process_keypoint_matching(
            outputs, img_sizes, threshold=threshold
        )

        return processed_outputs

    def plot_samples(self, processed_outputs, imgs):
        return self.processor.visualize_keypoint_matching(imgs, processed_outputs)


def align_images_after_superglue(img_1, img_2, matched):
    src_pts = matched["keypoints1"].cpu().numpy()
    dst_pts = matched["keypoints0"].cpu().numpy()

    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = dst_pts.reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=4.0
    )
    print(f"num inliers: {sum(mask)}")
    h, w = np.array(img_1).shape[:2]
    aligned = cv2.warpPerspective(np.array(img_2), H, (w, h))
    return aligned, H, mask

def mesh_px_to_render_px(mesh_x, mesh_y, center, mag):
    render_x = -(mesh_x - center[0]) * (512 / mag) + 512
    render_y = -(mesh_y - center[1]) * (512 / mag) + 512
    return render_x, render_y

def rotate_pix(render_x, render_y, rot_angle):
    theta = np.radians(rot_angle)    
    c, s = np.cos(theta), np.sin(theta)
    
    R = np.array([[c, -s], [s, c]])

    pt_rotated = R @ np.array([render_x - 512, render_y - 512])

    return pt_rotated[0] + 512, pt_rotated[1] + 512

def render_to_sat_pix(render_x, render_y, H):
    pt_sat = H @ np.array([render_x, render_y, 1])
    pt_sat = pt_sat / pt_sat[2]

    return pt_sat[0], pt_sat[1]

def sat_pixel_to_latlon(sat_x, sat_y, ds, orig_crs):
    cx = np.interp(sat_x, np.arange(len(ds.x)), ds.x.values)
    cy = np.interp(sat_y, np.arange(len(ds.y)), ds.y.values)

    if orig_crs != 4326:
        from pyproj import Transformer
        transformer = Transformer.from_crs(orig_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(cx, cy)
    else:
        lon, lat = cx, cy
    
    return lat, lon

def mesh_xy_to_latlon(mesh_x, mesh_y, center, mag, best_rot, H, ds):
    render_x, render_y = mesh_px_to_render_px(mesh_x, mesh_y, center, mag)
    render_x, render_y = rotate_pix(render_x, render_y, best_rot)
    sat_x, sat_y = render_to_sat_pix(render_x, render_y, H)
    lat, lon = sat_pixel_to_latlon(sat_x, sat_y, ds)
    return lat, lon
    
def load_mesh_and_combine(mesh_path):
    scene = trimesh.load(mesh_path)
    geometry_names = list(scene.geometry.keys())
    combined_trimesh = trimesh.util.concatenate([scene.geometry[name] for name in geometry_names])
    return combined_trimesh

def save_georeg(mesh_path, H, best_rot, ds, output_path, render_size = 1024):
    """
    Save a georegister mapping: for every mesh vertex, compute and store its lat/lon.

    Output .safetensors contains:
        world_points : (N, 3) float32 — mesh vertices in local 3D space
        latlon       : (N, 2) float32 — corresponding (lat, lon) on Earth

    Args:
        combined_trimesh : trimesh.Trimesh — the combined mesh
        H                : np.ndarray (3, 3) — homography from rotated render → satellite image
        best_rot         : float — rotation angle (degrees) applied before matching
        ds               : xarray.Dataset — NAIP dataset reprojected to EPSG:4326
        output_path      : str — path to save .safetensors file
        render_size      : int — size of the render image
    """

    assert ds.rio.crs, "the ds mush have a valid crs, use rio.write_crs to set it"
    combined_trimesh = load_mesh_and_combine(mesh_path)
    offset = render_size // 2
    center = combined_trimesh.centroid
    mag = max(combined_trimesh.extents[:2]) / 2.0
    vertices = combined_trimesh.vertices  # (N, 3)

    # mesh XY → render pixel
    render_x = -(vertices[:, 0] - center[0]) * (offset / mag) + offset
    render_y = -(vertices[:, 1] - center[1]) * (offset / mag) + offset

    # render pixel → rotated render pixel
    theta = np.radians(best_rot)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = render_x - offset, render_y - offset
    render_x = c * dx - s * dy + offset
    render_y = s * dx + c * dy + offset

    # rotated render pixel → satellite pixel (vectorized homography)
    pts = np.stack([render_x, render_y, np.ones(len(vertices))], axis=1)  # (N, 3)
    sat_pts = (H @ pts.T).T                                                # (N, 3)
    sat_x = sat_pts[:, 0] / sat_pts[:, 2]
    sat_y = sat_pts[:, 1] / sat_pts[:, 2]

    # satellite pixel → lat/lon
    lons = np.interp(sat_x, np.arange(len(ds.x)), ds.x.values)
    lats = np.interp(sat_y, np.arange(len(ds.y)), ds.y.values)

    if ds.rio.crs != 4326:
        from pyproj import Transformer
        transformer = Transformer.from_crs(ds.rio.crs.to_epsg(), "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(lons, lats)
    
    save_file(
        {
            "world_points": torch.from_numpy(vertices).to(torch.float32),
            "latlon": torch.tensor(np.stack([lats, lons], axis=1), dtype=torch.float32),
        },
        output_path,
    )
    print(f"[georeg] Saved {len(vertices):,} vertices to {output_path}")


def load_georeg(path):
    """Returns world_points (N, 3) and latlon (N, 2) tensors."""
    data = load_file(path)
    return data["world_points"], data["latlon"]

