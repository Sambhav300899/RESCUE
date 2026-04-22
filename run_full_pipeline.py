import argparse
import gc
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.colors import ListedColormap
from PIL import Image

from rescue import models, naip, utils, img_match
from rescue.feature_reduction import TorchIncrementalPCA
from rescue.lang_features import LSegLangFeatures
from rescue.mapanything_pipeline import (
    run_mapanything,
    save_language_features,
    save_mesh_to_glb,
    save_points_to_glb,
)

def extract_language_features(predictions, lang_feats, n_components=64):
    img_list = [pred['img_no_norm'] for pred in predictions]
    ipca = TorchIncrementalPCA(n_components=n_components, device='cuda')
    img_feats = []

    for img in tqdm.tqdm(img_list, desc="Extracting dense features"):
        img_perm = torch.permute(img, (0, 3, 1, 2))
        img_resized = torch.nn.functional.interpolate(
            img_perm, size=(480, 640), mode="bilinear", align_corners=False
        )
        img_feat = lang_feats.extract_dense_from_tensor(img_resized)
        img_feat = torch.nn.functional.interpolate(
            img_feat, size=(img.shape[1], img.shape[2]), mode="bilinear", align_corners=False
        )
        # (1, C, H, W) → (H, W, C)
        img_feat = torch.permute(img_feat.squeeze(0), (1, 2, 0))
        img_feats.append(img_feat)

    for img_feat in tqdm.tqdm(img_feats, desc="Fitting IPCA"):
        ipca.partial_fit(img_feat.view(-1, img_feat.shape[-1]))

    language_features = []
    for img_feat in tqdm.tqdm(img_feats, desc="Transforming with IPCA"):
        shape_hw = img_feat.shape[:2]
        flat_feat = img_feat.view(-1, img_feat.shape[-1])
        transformed = ipca.transform(flat_feat)
        transformed = transformed.view(*shape_hw, -1)
        language_features.append(transformed)

    language_features = torch.stack(language_features, dim=0)

    return language_features, ipca


robot2keywords = {
'spot': ['road', 'pavement', 'sidewalk', 'concrete', 'asphalt', 'building', 'grass', 'dirt', 'barren', 'concrete'],
'small drone': ['building', 'canopy', 'trees', 'forest', 'roof', 'balcony', 'overhang', 'awning'],
'large drone': ['water', 'pond', 'lake', 'river'],
}

def plot_robot2traverse(robot2traverse, img_rgb, run_dir):
    fig, axes = plt.subplots(nrows=1, ncols=len(robot2traverse), figsize=(20, 10))
    green_cmap = ListedColormap(['lime'])

    # plt.subplots returns a single Axes when ncols == 1; normalize to a list
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i, (robot, mask) in enumerate(robot2traverse.items()):
        axes[i].imshow(img_rgb)
        transparent_mask = np.ma.masked_where(mask == 0, mask)
        axes[i].imshow(transparent_mask, cmap=green_cmap, alpha=0.5)
        axes[i].set_title(robot)
        axes[i].axis('off')
    plt.suptitle("Traversability Maps", y = 0.8)
    fig.savefig(os.path.join(run_dir, "robot2traverse.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--sam3_model_path", type=str, default="generated/sam3")
    parser.add_argument("--run_dir", type=str, default="pipeline_run")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite outputs in run_dir (otherwise reuse intermediates when present).",
    )
    parser.add_argument("--add_language_features", type=ast.literal_eval, default=True)
    parser.add_argument("--use_ges", type=ast.literal_eval, default=True)
    parser.add_argument("--lseg_model_path", type=str, default="generated/lseg_minimal_e200.ckpt")
    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--dense_geospatial_align", type=ast.literal_eval, default=True)

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    sam3_model_path = args.sam3_model_path
    vid_path = args.video_path
    run_dir = args.run_dir
    overwrite = bool(args.overwrite)

    recon_pts_path = os.path.join(run_dir, "reconstruction_points.glb")
    recon_mesh_path = os.path.join(run_dir, "reconstruction_mesh.glb")
    recon_topdown_path = os.path.join(run_dir, "reconstruction_3d_topdown.png")
    lang_feats_path = os.path.join(run_dir, "language_features.safetensors")
    georeg_path = os.path.join(run_dir, "georeg.safetensors")

    ges_path = vid_path.replace(".mp4", ".json")
    geojson_path = vid_path.replace(".mp4", "_loc.geojson")

    print ('Downloading satellite imagery for GeoJSON AOI...')
    img_xr = naip.download_naip_for_geojson(
    geojson_path,
    "2018-01-01/2027-01-01", 
    )

    if img_xr is None:
        print("No NAIP imagery found for the given GeoJSON AOI")
        return
        
    else:
        print ('Satellite imagery downloaded successfully')
        img_rgb = np.moveaxis(img_xr.values[:3], 0, -1)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        ax.axis("off")
        fig.savefig(os.path.join(run_dir, "satellite_image.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)


    sam3_pred = models.sam3_predictor(sam3_model_path, device = 'cuda')

    robot2traverse = {}
    print ('Running SAM3 predictions for each robot...')
    for robot, keywords in robot2keywords.items():
        results = sam3_pred.pred_on_prompts_and_single_img(img_rgb, keywords, threshold=0.3, mask_threshold=0.5)
        masks, bboxes, scores, labels = utils.collate_sam3_results(results, keywords)

        robot2traverse[robot] = masks.max(axis = 0)

    plot_robot2traverse(robot2traverse, img_rgb, run_dir)
    print ('Traversability Maps plotted successfully')

    # Unload the sam3_pred model to free up GPU memory
    del sam3_pred
    gc.collect()
    torch.cuda.empty_cache()

    predictions = None
    if (not overwrite) and os.path.exists(recon_pts_path) and os.path.exists(recon_mesh_path):
        print(
            f"Found existing reconstruction intermediates; skipping MapAnything (use --overwrite to recompute): "
            f"{recon_pts_path}, {recon_mesh_path}"
        )
    else:
        print('Running MapAnything pipeline...')
        predictions = run_mapanything(vid_path, ges_path, fps=2, use_ges=args.use_ges)
        torch.cuda.empty_cache()

        save_points_to_glb(predictions, recon_pts_path, show_cam=False)
        save_mesh_to_glb(predictions, recon_mesh_path, show_cam=False)
        print('Reconstruction points and mesh saved successfully')

    # Reuse cached top-down render unless overwriting. When we do render, save the
    # raw render output directly (avoid matplotlib padding/cropping changing pixels).
    if (not overwrite) and os.path.exists(recon_topdown_path):
        render_3d = np.array(Image.open(recon_topdown_path))
    else:
        render_3d, _ = utils.render_3d_plot_from_above(
            recon_path=recon_mesh_path, bg_color=[0, 0, 0, 0]
        )
        Image.fromarray(render_3d).save(recon_topdown_path)

    if args.add_language_features:
        if (not overwrite) and os.path.exists(lang_feats_path):
            print(f"Found existing language features; skipping (use --overwrite to recompute): {lang_feats_path}")
        elif predictions is None:
            print(
                "Skipping language features because MapAnything predictions were not computed in this run "
                f"(delete {recon_pts_path}/{recon_mesh_path} or pass --overwrite to recompute)."
            )
        else:
            print ('Adding language features...')
            lang_feats = LSegLangFeatures(checkpoint_path=args.lseg_model_path)
            language_features, ipca = extract_language_features(predictions, lang_feats, args.n_components)
            print ('Language features extracted successfully')

            save_language_features(predictions, language_features, lang_feats_path, ipca)
            print ('Language features saved successfully')

            del language_features
            del lang_feats
            gc.collect()
            torch.cuda.empty_cache()

    if args.dense_geospatial_align and img_xr is not None:
        if (not overwrite) and os.path.exists(georeg_path):
            print(f"Found existing georegistration; skipping (use --overwrite to recompute): {georeg_path}")
            return
        print ('Co-registering 3D points with satellite imagery...')
        img_sat = Image.fromarray(img_rgb)
        img_3d = Image.fromarray(render_3d)
        rotation_angles = list(range(0, 360, 45))
        imgs_rotated = {x: img_3d.rotate(x) for x in rotation_angles}

        superglue_matcher = img_match.SuperGlueMatcher()


        matches = {}
        for rot, img in imgs_rotated.items():
            matches[rot] = superglue_matcher.match(img_sat, img)[0]
            print(f"num mactches for {rot} matches: {len(matches[rot]['matching_scores'])}")

        best_rot, best_matches = max(matches.items(), key=lambda x: len(x[1]['matching_scores']))
        vis_matches = superglue_matcher.plot_samples(processed_outputs=[best_matches], imgs = [img_rgb, imgs_rotated[best_rot]])[0]
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.axis("off")
        ax.imshow(vis_matches)
        ax.set_title(f"matches visualised, for best rotation of {best_rot}")
        fig.savefig(os.path.join(run_dir, "matches_visualised.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

        print ('3D points co-registered with satellite imagery successfully')

        aligned, H, mask = img_match.align_images_after_superglue(img_rgb, imgs_rotated[best_rot], best_matches)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.axis("off")
        ax.imshow(aligned)
        ax.set_title(f"aligned image, for best rotation of {best_rot}")
        fig.savefig(os.path.join(run_dir, "aligned_image.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

        img_match.save_georeg(recon_mesh_path, H, best_rot, img_xr, georeg_path)
        print ('Georegistration saved successfully')

if __name__ == "__main__":
    main()