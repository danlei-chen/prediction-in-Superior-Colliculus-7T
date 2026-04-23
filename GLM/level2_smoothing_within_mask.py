#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import glob
import subprocess

SMOOTH_FWHM_MM = [3]
MASK_FILES = [Path("/data/masks/sc_mask.nii.gz")]
INPUT_ROOT = Path("/data/level1_results")
PROJECTS = ['vision', 'somatosensory']
WARP_SPACE = 'mni'
ROI_NAMES = ['sc']

def find_input_maps(project, warp_space):
    pattern = INPUT_ROOT / project / warp_space / "sub-*" / "*cope*.nii.gz"
    files = glob.glob(str(pattern))
    return sorted([f for f in files if "smooth" not in f])

def make_output_name(input_file, smooth_mm, roi_name):
    input_file = Path(input_file)
    stem = input_file.name.replace(".nii.gz", "")
    new_name = f"{stem}_smooth{smooth_mm}mm_within{roi_name}mask.nii.gz"
    return input_file.parent / new_name

def smooth_within_mask(input_file, mask_file, output_file, smooth_mm):
    cmd = ["3dBlurInMask", "-input", str(input_file), "-mask", str(mask_file), 
           "-FWHM", str(smooth_mm), "-prefix", str(output_file), "-overwrite"]
    subprocess.run(cmd, check=True)

for project in PROJECTS:
    input_maps = find_input_maps(project, WARP_SPACE)
    for smooth_mm in SMOOTH_FWHM_MM:
        for roi_name, mask_file in zip(ROI_NAMES, MASK_FILES):
            for input_file in input_maps:
                output_file = make_output_name(input_file=input_file, smooth_mm=smooth_mm, roi_name=roi_name)
                smooth_within_mask(input_file=input_file, mask_file=mask_file, 
                                 output_file=output_file, smooth_mm=smooth_mm)