#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:25:00 2021

@author: chendanlei
"""
#python3 '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/scripts/subject_roi_mean_signal_RT_voxel.py'

import nibabel as nib
import numpy as np
import glob
from nilearn.glm import threshold_stats_img
from nilearn.image import resample_img
import pandas as pd 
import os
import statistics

#subject level roi directory
# task = 'emo'
# data_dir = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/results/level1//emoAvd_CS+-ActPasUSnegneu_motor/SC_warp/'
task = 'pain'
data_dir = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/results/level1/painAvd_CS+-ActPasUS1snegneu_motor/SC_warp/'
sc_distance_info = pd.read_csv('/Volumes/GoogleDrive/My Drive/U01/decision_making/roi/segment_SC/sc_segment_info_cope_space_'+task+'.csv')
# roi_dir = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/roi/subject_SC_mask/'+task+'Avd/warped_SC/'
# roi_thresh = 0.25
output = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_'+task+'_RT_voxel_distance.csv'
# file_format1 = 'cope2_smoothed3mm_masked.nii.gz'
# file_format2 = 'cope3_smoothed3mm_masked.nii.gz'
file_format1 = 'cope6.nii.gz'
file_format1_name = 'active_motor_RT'
# template_file = '/Volumes/GoogleDrive/My Drive/fMRI/atlas/MNI/mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_nlin_asym_09b/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii'

subj_list = [i.split('/')[-1] for i in glob.glob(data_dir+'*')]
subj_list.sort()

df = pd.DataFrame(columns = ['subjID', 'subject','run','task','type','x','y','z','signal','distance','side','quartile'])
for s in subj_list:
    print(s)
    
    subj = s[0:7]
    run = s[8:]

    try: 
        ######## 1 #######
        #load contrast
        file = glob.glob(data_dir+s+'/*'+file_format1)[0]
        file_img = nib.load(file)
        file_data = file_img.get_fdata()
        
        # #load roi
        # # roi_file = glob.glob(roi_dir+'/*'+s+'*')[0]
        # roi_file = glob.glob(roi_dir+'*'+s+'*')[0]
        # roi_data = nib.load(roi_file).get_fdata()    
        # # roi_data = resample_img(nib.load(roi_file),
        # #     target_affine=file_img.affine,
        # #     target_shape=file_img.shape[0:3],
        # #     interpolation='nearest').get_fdata()    
        # roi_data[roi_data<roi_thresh] = np.nan
        # roi_data[pd.notnull(roi_data)] = 1
        
        voxels_in_roi = (sc_distance_info['x'],sc_distance_info['y'],sc_distance_info['z'])
        for vox_n in range(len(voxels_in_roi[0])):
            # print(vox_n)
            x=voxels_in_roi[0][vox_n]; y=voxels_in_roi[1][vox_n]; z=voxels_in_roi[2][vox_n]
            
            #write to df
            df_row = pd.Series([s, subj, run, task, file_format1_name, x, y, z, file_data[x,y,z], 
                           sc_distance_info['distance'][(sc_distance_info['x']==x) & (sc_distance_info['y']==y) & (sc_distance_info['z']==z)][vox_n],                           
                           sc_distance_info['side'][(sc_distance_info['x']==x) & (sc_distance_info['y']==y) & (sc_distance_info['z']==z)][vox_n],                           
                           sc_distance_info['quartile'][(sc_distance_info['x']==x) & (sc_distance_info['y']==y) & (sc_distance_info['z']==z)][vox_n]], 
                           index = df.columns)
            df = df.append(df_row, ignore_index=True)
    
    except:
        print('no no_motor')
        
    df.to_csv(output)
    
    
    