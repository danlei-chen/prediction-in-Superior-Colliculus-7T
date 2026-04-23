import os
import nibabel as nib
import numpy as np
import pandas as pd

ROI_MASK_FILE = "/data/masks/sc_mask.nii.gz"
TEMPLATE_FILE = "/data/masks/template_mask.nii.gz"
OUTPUT_DIR = "/output/layer_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

roi_data = nib.load(ROI_MASK_FILE).get_fdata()
template_img = nib.load(TEMPLATE_FILE)

roi_data_right = roi_data.copy()
roi_data_left = roi_data.copy()
roi_data_right[: int(roi_data.shape[0] / 2), :, :] = np.nan
roi_data_left[int(roi_data.shape[0] / 2) :, :, :] = np.nan

roi_data_dist_right = np.full(roi_data.shape, np.nan)
roi_data_dist_left = np.full(roi_data.shape, np.nan)

right_center = np.array([101, 104, 62])
left_center = np.array([95, 104, 62])

right_xyz = np.where(roi_data_right == 1)
for x, y, z in zip(right_xyz[0], right_xyz[1], right_xyz[2]):
    roi_data_dist_right[x, y, z] = np.linalg.norm(np.array([x, y, z]) - right_center)

left_xyz = np.where(roi_data_left == 1)
for x, y, z in zip(left_xyz[0], left_xyz[1], left_xyz[2]):
    roi_data_dist_left[x, y, z] = np.linalg.norm(np.array([x, y, z]) - left_center)

roi_data_dist = roi_data_dist_left.copy()
roi_data_dist[~np.isnan(roi_data_dist_right)] = roi_data_dist_right[~np.isnan(roi_data_dist_right)]

valid_dist = roi_data_dist[~np.isnan(roi_data_dist)]
dist_scaled = (roi_data_dist - np.min(valid_dist)) / (np.max(valid_dist) - np.min(valid_dist))
all_dist = np.sort(np.unique(np.round(dist_scaled[~np.isnan(dist_scaled)], 3)))
distance_sort_bin = np.ceil((np.arange(1, len(all_dist) + 1) * 10) / len(all_dist)).astype(int)
distance_to_bin = dict(zip(all_dist, distance_sort_bin))

roi_data_bin = np.full(roi_data.shape, np.nan)
coords = np.where(~np.isnan(dist_scaled))
for x, y, z in zip(coords[0], coords[1], coords[2]):
    this_dist = np.round(dist_scaled[x, y, z], 3)
    roi_data_bin[x, y, z] = distance_to_bin[this_dist]

rows = []
coords = np.where(~np.isnan(roi_data_bin))
for x, y, z in zip(coords[0], coords[1], coords[2]):
    if x < int(roi_data.shape[0] / 2):
        side = "left"
    elif x > int(roi_data.shape[0] / 2):
        side = "right"

    rows.append(
        {"x": x,"y": y,"z": z,"side": side,
            "distance": roi_data_dist[x, y, z],
            "distance_scaled": np.round(dist_scaled[x, y, z], 3),
            "sc_layers": roi_data_bin[x, y, z]})

roi_info_df = pd.DataFrame(rows)
roi_info_df.to_csv(os.path.join(OUTPUT_DIR, "sc_segment_info.csv"), index=False)