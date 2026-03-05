import torch
import tqdm
import xarray as xr
import numpy as np
import cv2


def get_png_from_naip(naip_path):
    sample_img = xr.open_dataset(naip_path)

    rgb = sample_img.to_array()[1].values[:3, :, :]
    rgb = np.moveaxis(rgb, 0, -1).astype("uint8")

    return rgb


def plot_sam3_detections(rgb, masks, bboxes, scores, labels):
    """
    Plots SAM 3 detections on the RGB image using OpenCV.

    Args:
        rgb (np.ndarray): Original image (H, W, 3).
        masks (torch.Tensor or np.ndarray): Boolean masks (N, H, W).
        bboxes (torch.Tensor or np.ndarray): Bounding boxes (N, 4) in [x1, y1, x2, y2].
        scores (torch.Tensor or np.ndarray): Confidence scores (N,).
        labels (list of str): Class labels for each detection.
    """
    # Convert to numpy if necessary
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    output = rgb.copy()

    # Unique labels and their colors
    unique_labels = sorted(list(set(labels)))
    np.random.seed(42)
    label_colors = {
        label: np.random.randint(0, 255, 3, dtype=np.uint8).tolist()
        for label in unique_labels
    }

    # 1. Group and draw combined masks by label
    mask_overlay = output.copy()
    for label in unique_labels:
        # Get all masks for this label
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue

        # Combine masks for this label
        combined_label_mask = np.logical_or.reduce([masks[i] for i in indices])

        # Draw this label's combined mask
        mask_overlay[combined_label_mask] = label_colors[label]

    # Apply weighted blending for all masks at once
    cv2.addWeighted(mask_overlay, 0.4, output, 0.6, 0, output)

    # 2. Draw individual bounding boxes and labels
    for i, (bbox, score, label) in tqdm.tqdm(
        enumerate(zip(bboxes, scores, labels)), total=len(bboxes), desc="Drawing boxes"
    ):
        color = label_colors[label]

        # # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        # cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw label background
        cv2.rectangle(
            output, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1
        )
        # Draw text
        cv2.putText(
            output,
            text,
            (x1, y1 - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return output


def collate_sam3_results(results, prompts):
    labels = []
    masks = []
    bboxes = []
    scores = []

    for i, prompt in enumerate(prompts):
        if results[i]["scores"].shape[0] == 0:
            continue
        masks.append(results[i]["masks"].cpu())
        bboxes.append(results[i]["boxes"].cpu())
        scores.append(results[i]["scores"].cpu())
        labels.extend([prompt] * masks[-1].shape[0])

    masks = torch.cat(masks).numpy()
    bboxes = torch.cat(bboxes).numpy()
    scores = torch.cat(scores).numpy()

    return masks, bboxes, scores, labels
