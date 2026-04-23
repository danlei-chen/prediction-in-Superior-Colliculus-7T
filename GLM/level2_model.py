#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import glob
import os
import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl

FLAMEO_RUN_MODE = "ols"
LEVEL1_ROOT = Path("/data/level1_results")
LEVEL2_WORK_ROOT = Path("/work/level2")
CONTRASTS = ['cope1', 'cope2', 'cope3']
WARP_SPACE = 'mni'
SMOOTHING_LABELS = ['smooth3mm', 'nosmooth']
PROJECTS = ['vision', 'somatosensory']

fixed_fx = pe.Workflow(name="fixedfx")

copemerge = pe.MapNode(interface=fsl.Merge(dimension="t"), iterfield=["in_files"], name="copemerge")
level2model = pe.Node(interface=fsl.L2Model(), name="l2model")

if FLAMEO_RUN_MODE == "flame1":
    varcopemerge = pe.MapNode(interface=fsl.Merge(dimension="t"), iterfield=["in_files"], name="varcopemerge")

if FLAMEO_RUN_MODE == "ols":
    flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode="ols"), name="flameo", iterfield=["cope_file"])
    fixed_fx.connect([
        (copemerge, flameo, [("merged_file", "cope_file")]),
        (level2model, flameo, [("design_mat", "design_file"), ("design_con", "t_con_file"), ("design_grp", "cov_split_file")]),
    ])
elif FLAMEO_RUN_MODE == "flame1":
    flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode="flame1"), name="flameo", iterfield=["cope_file", "var_cope_file"])
    fixed_fx.connect([
        (copemerge, flameo, [("merged_file", "cope_file")]),
        (varcopemerge, flameo, [("merged_file", "var_cope_file")]),
        (level2model, flameo, [("design_mat", "design_file"), ("design_con", "t_con_file"), ("design_grp", "cov_split_file")]),
    ])

def build_analysis_mask(image_files, output_file):
    mask_data = np.mean(np.array([nib.load(f).get_fdata() for f in image_files]), axis=0)
    mask_data[mask_data != 0] = 1
    reference_img = nib.load(image_files[0])
    mask_img = nib.Nifti1Image(mask_data, reference_img.affine, reference_img.header)
    nib.save(mask_img, str(output_file))
    return output_file

def run_flameo(contrast, work_dir, path_pattern, group_level, smoothing_label):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / contrast).mkdir(exist_ok=True)
    fixed_fx.base_dir = str(work_dir / contrast)
    
    cope_pattern = path_pattern + f"*_{contrast}*"
    if FLAMEO_RUN_MODE == "flame1":
        varcope_pattern = path_pattern + f"*var{contrast}_*"
    
    cope_files = sorted(glob.glob(cope_pattern))
    
    if not group_level:
        if smoothing_label != "no_smoothing":
            cope_files = [f for f in cope_files if smoothing_label in f]
        else:
            cope_files = [f for f in cope_files if "smooth" not in f]
    
    if len(cope_files) == 0:
        raise RuntimeError(f"No cope files found for {contrast}")
    
    mask_file = build_analysis_mask(image_files=cope_files, output_file=work_dir / f"mask_{contrast}.nii.gz")
    fixed_fx.inputs.flameo.mask_file = str(mask_file)
    fixed_fx.inputs.copemerge.in_files = cope_files
    fixed_fx.inputs.l2model.num_copes = len(cope_files)
    
    if FLAMEO_RUN_MODE == "flame1":
        varcope_files = sorted(glob.glob(varcope_pattern))
        if not group_level and smoothing_label != "no_smoothing":
            varcope_files = [f for f in varcope_files if smoothing_label in f]
        fixed_fx.inputs.varcopemerge.in_files = varcope_files
    
    fixed_fx.run()

group_level = False
for project in PROJECTS:
    for smoothing_label in SMOOTHING_LABELS:
        subject_dirs = glob.glob(str(LEVEL1_ROOT / project / WARP_SPACE / "sub*"))
        subject_dirs.sort()
        for subject_id in subject_dirs:
            for contrast in CONTRASTS:
                print(f"{project} | {smoothing_label} | {subject_id} | {contrast}")
                work_dir = LEVEL2_WORK_ROOT / f"{project}_{smoothing_label}_{subject_id}"
                path_pattern = str(LEVEL1_ROOT / project / WARP_SPACE / f"{subject_id}*" / "")
                run_flameo(contrast=contrast, work_dir=work_dir, path_pattern=path_pattern, 
                          group_level=group_level, smoothing_label=smoothing_label)

group_level = True
for project in PROJECTS:
    for smoothing_label in SMOOTHING_LABELS:
        subject_workdirs = glob.glob(str(LEVEL2_WORK_ROOT / f"{project}_{smoothing_label}_sub*"))
        subject_workdirs.sort()
        for contrast in CONTRASTS:
            print(f"GROUP: {project} | {smoothing_label} | {contrast}")
            work_dir = LEVEL2_WORK_ROOT / f"{project}_{smoothing_label}"
            path_pattern = str(LEVEL2_WORK_ROOT / f"{project}_{smoothing_label}_sub*" / contrast / 
                             "fixedfx" / "flameo" / "mapflow" / "_flameo0" / "stats" / "")
            run_flameo(contrast=contrast, work_dir=work_dir, path_pattern=path_pattern,
                      group_level=group_level, smoothing_label=smoothing_label)