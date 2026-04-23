#!/usr/bin/env python3

from pathlib import Path
import glob
import nibabel as nib
import pandas as pd

DATA_DIR = Path("/data/roi_space")
ROI_INFO_FILE = Path("/data/masks/sc_segment_info.csv")
OUTPUT_CSV = Path("/output/roi_signal_by_voxel.csv")

CONTRAST_FILES = [
    ("cope1.nii.gz", "shape"),
    ("cope2.nii.gz", "decision"),
    ("cope3.nii.gz", "stimulation")]

roi_info = pd.read_csv(ROI_INFO_FILE)

subject_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
rows = []

for subject_dir in subject_dirs:
    print(subject_dir.name)

    subject = subject_dir.name.split("_")[0]
    run = subject_dir.name.split("_")[1]

    for contrast_file, condition_name in CONTRAST_FILES:
        matches = glob.glob(str(subject_dir / f"*{contrast_file}"))
        if len(matches) == 0:
            continue

        image = nib.load(matches[0])
        image_data = image.get_fdata()

        for _, voxel in roi_info.iterrows():
            x = int(voxel["x"])
            y = int(voxel["y"])
            z = int(voxel["z"])

            rows.append({
                "subject_run": subject_dir.name,
                "subject": subject,"run": run,
                "task": "vision",
                "contrast_file": contrast_file,
                "condition": condition_name,
                "x": x,"y": y,"z": z,
                "signal": image_data[x, y, z],
                "distance": voxel["distance"],
                "side": voxel["side"],
                "sc_layer": voxel["sc_layer"]})

output_df = pd.DataFrame(rows)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)