#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:25:00 2021

@author: chendanlei
"""
#python3 '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/scripts/subject_roi_mean_signal.py'

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

roi_and_output_dirs = [('/Volumes/GoogleDrive/My Drive/fMRI/PAG_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/PAG_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Marta_LGN_all.nii.gzz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/LGN_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/fMRI/roi/Wager_atlas/subcortex_combined/Wager_atlas-Diencephalon-Thal_Hythal.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/Hypothalamus_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/fMRI/roi/Wager_atlas/subcortex_combined/Wager_atlas-Diencephalon-Thal_VPL.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/VPL_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/fMRI/roi/7Trep_Marta/Hippocampus_resammpled.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/Hippocampus_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_A1_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/A1_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_FEF_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/FEF_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_LIP_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/LIP_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_V1_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/V1_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Marta_MGN_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/MGN_signal_'+task+'.csv')]

roi_and_output_dirs = [('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_S1_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/S1_signal_'+task+'.csv'),
                       ('/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/roi/Glasser_M1_all.nii.gz', '/Volumes/GoogleDrive/My Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/M1_signal_'+task+'.csv')]

for roi_and_output_dir in roi_and_output_dirs:
    
    roi_dir = roi_and_output_dir[0]
    print(roi_dir)
    output = roi_and_output_dir[1]
    print(output)
    
    df = pd.DataFrame(columns = ['subjID', 'subject','run','task','type','roi_dim1','roi_size','mean_signal','peak_signal'])
    for s in subj_list:
        print(s)
        
        subj = s[0:7]
        run = s[8:]
            
        ######## 1 #######
        #load contrast
        file = glob.glob(data_dir+s+'*/*'+file_format1)[0]
        file_img = nib.load(file)
        file_data = file_img.get_fdata()
    
        #load roi
        roi_data = resample_img(nib.load(roi_dir),
            target_affine=file_img.affine,
            target_shape=file_img.shape[0:3],
            interpolation='nearest').get_fdata()    
        roi_data[roi_data!=1] = np.nan
        roi_data[pd.notnull(roi_data)] = 1
        roi_size = len(roi_data[pd.notnull(roi_data)])
        
        #mask contrast file
        file_data[np.isnan(roi_data)] = np.nan
        #get mean
        mean_signal = np.nanmean(file_data)
        #get peak
        max_signal = np.nanmax(file_data)
        #write to df
        df_row = pd.Series([s, subj, run, task, file_format1_name, 'all', roi_size,mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
        
        #get left roi
        roi_data = resample_img(nib.load(roi_dir),
            target_affine=file_img.affine,
            target_shape=file_img.shape[0:3],
            interpolation='nearest').get_fdata()    
        roi_data[roi_data!=1] = np.nan
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
        df_row = pd.Series([s, subj, run, task, file_format1_name, 'left', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        #get right roi
        roi_data = resample_img(nib.load(roi_dir),
            target_affine=file_img.affine,
            target_shape=file_img.shape[0:3],
            interpolation='nearest').get_fdata()    
        roi_data[roi_data!=1] = np.nan
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
        df_row = pd.Series([s, subj, run, task, file_format1_name, 'right', roi_size, mean_signal, max_signal], index = df.columns)
        df = df.append(df_row, ignore_index=True)
    
        try: 
        
            ######## 2 #######
            #load contrast
            file = glob.glob(data_dir+s+'*/*'+file_format2)[0]
            file_img = nib.load(file)
            file_data = file_img.get_fdata()
            
            #load roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
            roi_data[pd.notnull(roi_data)] = 1
            roi_size = len(roi_data[pd.notnull(roi_data)])
            
            #mask contrast file
            file_data[np.isnan(roi_data)] = np.nan
            #get mean
            mean_signal = np.nanmean(file_data)
            #get peak
            max_signal = np.nanmax(file_data)
            #write to df
            df_row = pd.Series([s, subj, run, task, file_format2_name, 'all', roi_size,mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
            
            #get left roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
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
            df_row = pd.Series([s, subj, run, task, file_format2_name, 'left',roi_size, mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
        
            #get right roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
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
            df_row = pd.Series([s, subj, run, task, file_format2_name, 'right', roi_size, mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
        
        except:
            print('no no_motor')
            
        try: 
        
            ######## 3 #######
            #load contrast
            file = glob.glob(data_dir+s+'*/*'+file_format3)[0]
            file_img = nib.load(file)
            file_data = file_img.get_fdata()
            
            #load roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
            roi_data[pd.notnull(roi_data)] = 1
            roi_size = len(roi_data[pd.notnull(roi_data)])
            
            #mask contrast file
            file_data[np.isnan(roi_data)] = np.nan
            #get mean
            mean_signal = np.nanmean(file_data)
            #get peak
            max_signal = np.nanmax(file_data)
            #write to df
            df_row = pd.Series([s, subj, run, task, file_format3_name, 'all', roi_size,mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
            
            #get left roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
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
            df_row = pd.Series([s, subj, run, task, file_format3_name, 'left',roi_size, mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
        
            #get right roi
            roi_data = resample_img(nib.load(roi_dir),
                target_affine=file_img.affine,
                target_shape=file_img.shape[0:3],
                interpolation='nearest').get_fdata()    
            roi_data[roi_data!=1] = np.nan
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
            df_row = pd.Series([s, subj, run, task, file_format3_name, 'right', roi_size, mean_signal, max_signal], index = df.columns)
            df = df.append(df_row, ignore_index=True)
        
        except:
            print('no motor')
            
        df.to_csv(output)
        
        
        