#!/usr/bin/env python3
"""
Read a GES-style JSON (cameraFrames[].coordinate lat/lon) and write a GeoJSON
FeatureCollection with one Polygon = axis-aligned bounding box in WGS84.

By default, expands the box using focal length + image size + altitude so a flat,
nadir ground plane would stay inside the bbox for every frame (see
--footprint-factor for oblique / safety margin).

Example:
  python scripts/ges_json_to_bounds_geojson.py path/to/scene.json
  python scripts/ges_json_to_bounds_geojson.py scene.json -o scene_bounds.geojson --pad-deg 0.0002
  python scripts/ges_json_to_bounds_geojson.py scene.json --no-frustum-pad
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ~ meters per degree latitude; lon scale uses cos(latitude) below
M_PER_DEG_LAT = 111_320.0


def focal_length_px(h_px: float, fov_vertical_deg: float) -> float:
    """Match src/rescue/ges_utils.py intrinsics."""
    return h_px / (2.0 * math.tan(math.radians(fov_vertical_deg / 2.0)))


def footprint_radius_m_nadir(
    alt_m: float,
    w_px: float,
    h_px: float,
    fov_vertical_deg: float,
) -> float:
    """
    Approx. horizontal distance on flat ground from nadir to the projection of the
    farthest image corner, using a pinhole model (same fl as ges_utils).
    """
    if alt_m <= 0 or w_px <= 0 or h_px <= 0 or fov_vertical_deg <= 0:
        return 0.0
    fl = focal_length_px(h_px, fov_vertical_deg)
    if fl <= 0:
        return 0.0
    r_px = math.hypot(w_px / 2.0, h_px / 2.0)
    theta = math.atan2(r_px, fl)
    return alt_m * math.tan(theta)


def bounds_geojson_from_ges(
    data: dict,
    name: str | None = None,
    pad_deg: float = 0.0,
    *,
    footprint_factor: float = 1.25,
    use_frustum_pad: bool = True,
) -> dict:
    frames = data["cameraFrames"]
    if not frames:
        raise ValueError("cameraFrames is empty")

    w_px = float(data["width"]) if "width" in data else 0.0
    h_px = float(data["height"]) if "height" in data else 0.0
    can_frustum = use_frustum_pad and w_px > 0 and h_px > 0
    footprint_factor = max(0.0, float(footprint_factor))

    min_lat = math.inf
    max_lat = -math.inf
    min_lon = math.inf
    max_lon = -math.inf
    max_r_m = 0.0

    for fr in frames:
        c = fr["coordinate"]
        lat = float(c["latitude"])
        lon = float(c["longitude"])
        alt_m = float(c.get("altitude", 0.0))

        pad_lat_deg = 0.0
        pad_lon_deg = 0.0
        if can_frustum:
            fov_v = float(fr["fovVertical"])
            r_m = footprint_radius_m_nadir(alt_m, w_px, h_px, fov_v) * footprint_factor
            max_r_m = max(max_r_m, r_m)
            if r_m > 0:
                pad_lat_deg = r_m / M_PER_DEG_LAT
                clat = max(1e-6, math.cos(math.radians(lat)))
                pad_lon_deg = r_m / (M_PER_DEG_LAT * clat)

        min_lat = min(min_lat, lat - pad_lat_deg)
        max_lat = max(max_lat, lat + pad_lat_deg)
        min_lon = min(min_lon, lon - pad_lon_deg)
        max_lon = max(max_lon, lon + pad_lon_deg)

    if pad_deg:
        min_lat -= pad_deg
        max_lat += pad_deg
        min_lon -= pad_deg
        max_lon += pad_deg

    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]

    label = name or "ges_scene"
    props: dict = {
        "name": label,
        "frames": len(frames),
        "frustum_pad": bool(can_frustum),
        "footprint_factor": footprint_factor if can_frustum else None,
        "max_footprint_radius_m": round(max_r_m, 3) if can_frustum else None,
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="GES JSON → GeoJSON lat/lon bounding box")
    p.add_argument("input_json", type=Path, help="Path to GES JSON (with cameraFrames)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .geojson path (default: <input_stem>_bounds.geojson)",
    )
    p.add_argument(
        "--pad-deg",
        type=float,
        default=0.0,
        help="Extra expand: add this many degrees to each side after frustum pad (default: 0)",
    )
    p.add_argument(
        "--footprint-factor",
        type=float,
        default=1.25,
        help=(
            "Multiply nadir footprint radius (alt * tan(corner_half_angle)); "
            "use >1 for oblique views / margin (default: 1.25)"
        ),
    )
    p.add_argument(
        "--no-frustum-pad",
        action="store_true",
        help="Only use camera path lat/lon (ignore focal / FOV / altitude footprint)",
    )
    args = p.parse_args()

    inp = args.input_json
    if not inp.is_file():
        print(f"Error: not a file: {inp}", file=sys.stderr)
        return 1

    with inp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "cameraFrames" not in data:
        print("Error: JSON missing top-level key 'cameraFrames'", file=sys.stderr)
        return 1

    if not args.no_frustum_pad and ("width" not in data or "height" not in data):
        print(
            "Warning: missing width/height; frustum padding skipped (use --no-frustum-pad to silence)",
            file=sys.stderr,
        )

    out = args.output or inp.with_name(f"{inp.stem}_bounds.geojson")
    gj = bounds_geojson_from_ges(
        data,
        name=inp.stem,
        pad_deg=args.pad_deg,
        footprint_factor=args.footprint_factor,
        use_frustum_pad=not args.no_frustum_pad,
    )
    out.write_text(json.dumps(gj, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
