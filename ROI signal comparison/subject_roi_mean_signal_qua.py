#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:25:00 2021

@author: chendanlei
"""
#python3 '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/scripts/subject_roi_mean_signal_qua.py'

import nibabel as nib
import numpy as np
import glob
from nilearn.glm import threshold_stats_img
from nilearn.image import resample_img
import pandas as pd 
import os
import statistics

#subject level roi directory
task = 'emo'
data_dir = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/results/level1//emoAvd_CS+-ActPasUSnegneu_motor/SC_warp/'
# task = 'pain'
# data_dir = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/results/level1/painAvd_CS+-ActPasUS1snegneu_motor/SC_warp/'
roi_dir = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/roi/subject_SC_mask/'+task+'Avd/warped_SC/'
roi_thresh = 0.25
output = '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_'+task+'_'+str(roi_thresh)+'_qua.csv'
# file_format1 = 'cope2_smoothed3mm_masked.nii.gz'
# file_format2 = 'cope3_smoothed3mm_masked.nii.gz'
file_format1 = 'cope3.nii.gz'
file_format1_name = 'passive'
file_format2 = 'cope4.nii.gz'
file_format2_name = 'active_no_motor'
file_format3 = 'cope5.nii.gz'
file_format3_name = 'active_motor'
template_file = '/Volumes/GoogleDrive/My Drive/fMRI/atlas/MNI/mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_nlin_asym_09b/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii'

subj_list = [i.split('/')[-1] for i in glob.glob(data_dir+'*')]
subj_list.sort()

df = pd.DataFrame(columns = ['subjID', 'subject','run','task','type','roi_dim1','roi_dim3','roi_size','mean_signal','peak_signal'])
for s in subj_list:
    print(s)
    
    subj = s[0:7]
    run = s[8:]
    
    ######## 1 #######
    #load contrast
    file = glob.glob(data_dir+s+'/*'+file_format1)[0]
    file_img = nib.load(file)
    file_data = file_img.get_fdata()

    #load roi
    # roi_file = glob.glob(roi_dir+'/*SC*'+s+'*')[0]
    roi_file = glob.glob(roi_dir+'*'+s+'*')[0]
    roi_data = nib.load(roi_file).get_fdata()    
    # roi_data = resample_img(nib.load(roi_file),
    #     target_affine=file_img.affine,
    #     target_shape=file_img.shape[0:3],
    #     interpolation='nearest').get_fdata()    
    roi_data[roi_data<roi_thresh] = np.nan
    roi_data[pd.notnull(roi_data)] = 1
    roi_size = len(roi_data[pd.notnull(roi_data)])
    
    #mask contrast file
    file_data[np.isnan(roi_data)] = np.nan
    #get mean
    mean_signal = np.nanmean(file_data)
    #get peak
    max_signal = np.nanmax(file_data)
    #write to df
    df_row = pd.Series([s, subj, run, task, file_format1_name, 'all', 'all', roi_size,mean_signal, max_signal], index = df.columns)
    df = df.append(df_row, ignore_index=True)
    
    #get upper left roi
    roi_data = nib.load(roi_file).get_fdata()    
    # roi_data = resample_img(nib.load(roi_file),
    #     target_affine=file_img.affine,
    #     target_shape=file_img.shape[0:3],
    #     interpolation='nearest').get_fdata()    
    roi_data[roi_data<roi_thresh] = np.nan
    roi_data[pd.notnull(roi_data)] = 1
    roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left,
    height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
    roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
    roi_size = len(roi_data[pd.notnull(roi_data)])
    #mask contrast file
    file_data = nib.load(file).get_fdata()
    file_data[np.isnan(roi_data)] = np.nan
    # test_img = nib.Nifti1Image(file_data, nib.load(file).affine, nib.load(file).header)
    # nib.save(test_img, '/Users/chendanlei/Desktop/x.nii.gz')
    #get mean
    mean_signal = np.nanmean(file_data)
    #get peak
    max_signal = np.nanmax(file_data)
    #write to df
    df_row = pd.Series([s, subj, run, task, file_format1_name, 'left', 'upper', roi_size, mean_signal, max_signal], index = df.columns)
    df = df.append(df_row, ignore_index=True)

    #get lower left roi
    roi_data = nib.load(roi_file).get_fdata()    
    # roi_data = resample_img(nib.load(roi_file),
    #     target_affine=file_img.affine,
    #     target_shape=file_img.shape[0:3],
    #     interpolation='nearest').get_fdata()    
    roi_data[roi_data<roi_thresh] = np.nan
    roi_data[pd.notnull(roi_data)] = 1
    roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left,
    height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
    roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
    roi_size = len(roi_data[pd.notnull(roi_data)])
    #mask contrast file
    file_data = nib.load(file).get_fdata()
    file_data[np.isnan(roi_data)] = np.nan
    #get mean
    mean_signal = np.nanmean(file_data)
    #get peak
    max_signal = np.nanmax(file_data)
    #write to df
    df_row = pd.Series([s, subj, run, task, file_format1_name, 'left', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
    df = df.append(df_row, ignore_index=True)

    #get upper right roi
    roi_data = nib.load(roi_file).get_fdata()    
    # roi_data = resample_img(nib.load(roi_file),
    #     target_affine=file_img.affine,
    #     target_shape=file_img.shape[0:3],
    #     interpolation='nearest').get_fdata()    
    roi_data[roi_data<roi_thresh] = np.nan
    roi_data[pd.notnull(roi_data)] = 1
    roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
    height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
    roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
    roi_size = len(roi_data[pd.notnull(roi_data)])
    #mask contrast file
    file_data = nib.load(file).get_fdata()
    file_data[np.isnan(roi_data)] = np.nan
    #get mean
    mean_signal = np.nanmean(file_data)
    #get peak
    max_signal = np.nanmax(file_data)
    #write to df
    df_row = pd.Series([s, subj, run, task, file_format1_name, 'right', 'upper', roi_size, mean_signal, max_signal], index = df.columns)
    df = df.append(df_row, ignore_index=True)
    
    #get lower right roi
    roi_data = nib.load(roi_file).get_fdata()    
    # roi_data = resample_img(nib.load(roi_file),
    #     target_affine=file_img.affine,
    #     target_shape=file_img.shape[0:3],
    #     interpolation='nearest').get_fdata()    
    roi_data[roi_data<roi_thresh] = np.nan
    roi_data[pd.notnull(roi_data)] = 1
    roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
    height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
    roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
    roi_size = len(roi_data[pd.notnull(roi_data)])
    #mask contrast file
    file_data = nib.load(file).get_fdata()
    file_data[np.isnan(roi_data)] = np.nan
    #get mean
    mean_signal = np.nanmean(file_data)
    #get peak
    max_signal = np.nanmax(file_data)
    #write to df
    df_row = pd.Series([s, subj, run, task, file_format1_name, 'right', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
    df = df.append(df_row, ignore_index=True)

    try: 
        ######## 2 #######
        #load contrast
        file = glob.glob(data_dir+s+'/*'+file_format2)[0]
        file_img = nib.load(file)
        file_data = file_img.get_fdata()
        
        #load roi
        # roi_file = glob.glob(roi_dir+'/*'+s+'*')[0]
        roi_file = glob.glob(roi_dir+'*'+s+'*')[0]
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_size = len(roi_data[pd.notnull(roi_data)])
        
        #mask contrast file
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format2_name, 'all', 'all', roi_size,mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
        
        #get upper left roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format2_name, 'left', 'upper',roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get lower left roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format2_name, 'left', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get upper right roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format2_name, 'right', 'upper', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get lower right roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format2_name, 'right', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
    except:
        print('no no_motor')
        
    try:
        ######## 3 #######
        #load contrast
        file = glob.glob(data_dir+s+'/*'+file_format3)[0]
        file_img = nib.load(file)
        file_data = file_img.get_fdata()
        
        #load roi
        # roi_file = glob.glob(roi_dir+'/*'+s+'*')[0]
        roi_file = glob.glob(roi_dir+'*'+s+'*')[0]
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_size = len(roi_data[pd.notnull(roi_data)])
        
        #mask contrast file
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format3_name, 'all', 'all', roi_size,mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
        
        #get upper left roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format3_name, 'left', 'upper',roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get lower left roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[int(roi_data.shape[0]/2):,:,:] = np.nan #first dim second half - left
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format3_name, 'left', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get upper right roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,0:height_middle_split] = np.nan #third dim first half - upper,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format3_name, 'right', 'upper', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get lower right roi
        roi_data = nib.load(roi_file).get_fdata()    
        # roi_data = resample_img(nib.load(roi_file),
        #     target_affine=file_img.affine,
        #     target_shape=file_img.shape[0:3],
        #     interpolation='nearest').get_fdata()    
        roi_data[roi_data<roi_thresh] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_data[0:int(roi_data.shape[0]/2),:,:] = np.nan #first dim second half - right
        height_middle_split = int(statistics.median(np.where(~np.isnan(roi_data))[2]))
        roi_data[:,:,height_middle_split:] = np.nan #third dim second half - lower,
        roi_size = len(roi_data[pd.notnull(roi_data)])
        #mask contrast file
        file_data = nib.load(file).get_fdata()
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format3_name, 'right', 'lower', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)

    except:
        print('no motor')
        
    df.to_csv(output)
    
    
    