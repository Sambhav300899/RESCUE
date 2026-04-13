"""
Standalone script wrapper around rescue.mapanything_pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/run_mapanything.py \
        --video   generated/lawnmower.mp4 \
        --ges     generated/test1.json \
        --output  generated/reconstruction.glb \
        --fps     1
"""

import argparse
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from rescue.mapanything_pipeline import predict, save_glb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",           default="generated/lawnmower.mp4")
    parser.add_argument("--ges",             default="generated/test1.json")
    parser.add_argument("--output",          default="generated/reconstruction.glb")
    parser.add_argument("--fps",             type=float, default=1.0)
    parser.add_argument("--num_frames",      type=int,   default=None)
    parser.add_argument("--model_dir",       default="generated/map-anything")
    parser.add_argument("--temp_dir",        default="generated/temp")
    parser.add_argument("--conf_percentile", type=float, default=3.0)
    args = parser.parse_args()

    pred_dict = predict(
        video_path=args.video,
        ges_path=args.ges,
        fps=args.fps,
        num_frames=args.num_frames,
        model_dir=args.model_dir,
        temp_dir=args.temp_dir,
    )

    save_glb(
        pred_dict,
        output_path=args.output.replace(".glb", "only_mesh.glb"),
        conf_percentile=args.conf_percentile,
        show_cam=False,
        as_mesh=True,
    )

    del pred_dict['conf']
    save_glb(
    pred_dict,
    output_path=args.output.replace(".glb", "only_points.glb"),
    show_cam=True,
    as_mesh=False,       # True = mesh, False = point cloud
    conf_percentile=0,   # 0 = keep all points
    )