#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nibabel as nib
import numpy as np
import os
import glob
from nilearn.image import resample_img, mean_img, concat_imgs
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm

output_df = pd.DataFrame(columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])
svc = SVC(kernel='linear')

for k, test_subjects in enumerate(test_groups):
    for train_type, test_type in train_test_set:
        if train_type == 'stimulation':
            X_train = consequence_concat_img_data2D_masked[[i for i, s in enumerate(consequence_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([consequence_event_files['sensory_labels'][i] for i, s in enumerate(consequence_event_files['subject']) if s not in test_subjects])
        elif train_type == 'decision':
            X_train = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] != 'shape'], :]
            y_train = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] != 'shape'])
        elif train_type == 'shape':
            X_train = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] == 'shape'], :]
            y_train = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] == 'shape'])
        
        if test_type == 'stimulation':
            X_test = consequence_concat_img_data2D_masked[[i for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([consequence_event_files['sensory_labels'][i] for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects])
        elif test_type == 'decision':
            X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'], :]
            y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'])
        elif test_type == 'shape':
            X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'], :]
            y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'])
        
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = np.mean(y_pred == y_test)
        output_df = pd.concat([output_df, pd.DataFrame([[train_type, test_type, k, test_subjects, acc]], 
                                                        columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])], ignore_index=True)

subj_to_label = {}
for s, lab in zip(consequence_event_files['subject'], consequence_event_files['sensory_labels']):
    if s not in subj_to_label:
        subj_to_label[s] = lab

labels_present = sorted(list(set(subj_to_label.values())))
label_a, label_b = labels_present[0], labels_present[1]
subj_list = np.unique([i.split('/')[-1].split('wdata_')[1].split('_')[0] for i in consequence_all_files])

group_a = [s for s in subj_list if subj_to_label.get(s, None) == label_a]
group_b = [s for s in subj_list if subj_to_label.get(s, None) == label_b]

group_a = shuffle(np.array(group_a), random_state=0).tolist()
group_b = shuffle(np.array(group_b), random_state=1).tolist()

n_pairs = min(len(group_a), len(group_b))
test_groups = [[group_a[i], group_b[i]] for i in range(n_pairs)]
nfolds = len(test_groups)

train_test_set = [('decision', 'stimulation'), ('decision', 'decision'), ('stimulation', 'decision'),
                  ('stimulation', 'stimulation'), ('shape', 'stimulation'), ('shape', 'decision'),
                  ('stimulation', 'shape'), ('decision', 'shape'), ('shape', 'shape')]

output_df = pd.DataFrame(columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])
svc = SVC(kernel='linear')

for k, test_subjects in enumerate(test_groups):
    for train_type, test_type in train_test_set:
        if train_type == 'stimulation':
            X_train = consequence_concat_img_data2D_masked[[i for i, s in enumerate(consequence_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([consequence_event_files['sensory_labels'][i] for i, s in enumerate(consequence_event_files['subject']) if s not in test_subjects])
        elif train_type == 'decision':
            X_train = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] != 'shape'], :]
            y_train = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] != 'shape'])
        elif train_type == 'shape':
            X_train = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] == 'shape'], :]
            y_train = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] == 'shape'])
        
        if test_type == 'stimulation':
            X_test = consequence_concat_img_data2D_masked[[i for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([consequence_event_files['sensory_labels'][i] for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects])
        elif test_type == 'decision':
            X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'], :]
            y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'])
        elif test_type == 'shape':
            X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'], :]
            y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'])
        
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = np.mean(y_pred == y_test)
        output_df = pd.concat([output_df, pd.DataFrame([[train_type, test_type, k, test_subjects, acc]], 
                                                        columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])], ignore_index=True)

output_df.to_csv('mvpa_classification_paired_leave2out.csv', index=False)
print(f"Mean accuracy: {np.mean(output_df['accuracy']):.4f}")

summary_df = output_df.groupby(['train_type', 'test_type'], as_index=False)['accuracy'].agg(['mean', 'sem']).reset_index().rename(columns={'mean': 'mean_accuracy', 'sem': 'sem_accuracy'})

def permute_labels_by_subject(subjects_per_sample, y_labels, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    subjects_per_sample = np.asarray(subjects_per_sample)
    y_labels = np.asarray(y_labels)
    unique_subjects = np.unique(subjects_per_sample)
    subj_to_label = {}
    for subj in unique_subjects:
        lbls = y_labels[subjects_per_sample == subj]
        subj_to_label[subj] = lbls[0]
    orig_labels = np.array([subj_to_label[subj] for subj in unique_subjects])
    perm_labels = rng.permutation(orig_labels)
    subj_to_perm_label = dict(zip(unique_subjects, perm_labels))
    y_perm = np.array([subj_to_perm_label[subj] for subj in subjects_per_sample])
    return y_perm

n_permutations = 1000
start_permutation = 0
perm_df = pd.DataFrame(columns=['train_type', 'test_type', 'fold', 'accuracy', 'perm', 'test_subjects'])
svc = SVC(kernel='linear')

if os.path.exists('mvpa_classification_paired_leave2out_permutation_subjshuffle.csv'):
    perm_df = pd.read_csv('mvpa_classification_paired_leave2out_permutation_subjshuffle.csv')
    start_permutation = int(np.max(perm_df['perm'])) + 1

for perm in tqdm(range(start_permutation, n_permutations), desc="Permutation Test"):
    for k, test_subjects in enumerate(test_groups):
        for train_type, test_type in train_test_set:
            if train_type == 'stimulation':
                train_idx = [i for i, s in enumerate(consequence_event_files['subject']) if s not in test_subjects]
                X_train = consequence_concat_img_data2D_masked[train_idx, :]
                y_train = np.array([consequence_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([consequence_event_files['subject'][i] for i in train_idx])
            elif train_type == 'decision':
                train_idx = [i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] != 'shape']
                X_train = selection_concat_img_data2D_masked[train_idx, :]
                y_train = np.array([selection_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([selection_event_files['subject'][i] for i in train_idx])
            elif train_type == 'shape':
                train_idx = [i for i, s in enumerate(selection_event_files['subject']) if s not in test_subjects and selection_event_files['decision_labels'][i] == 'shape']
                X_train = selection_concat_img_data2D_masked[train_idx, :]
                y_train = np.array([selection_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([selection_event_files['subject'][i] for i in train_idx])
            
            if test_type == 'stimulation':
                X_test = consequence_concat_img_data2D_masked[[i for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects], :]
                y_test = np.array([consequence_event_files['sensory_labels'][i] for i, s in enumerate(consequence_event_files['subject']) if s in test_subjects])
            elif test_type == 'decision':
                X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'], :]
                y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] != 'shape'])
            elif test_type == 'shape':
                X_test = selection_concat_img_data2D_masked[[i for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'], :]
                y_test = np.array([selection_event_files['sensory_labels'][i] for i, s in enumerate(selection_event_files['subject']) if s in test_subjects and selection_event_files['decision_labels'][i] == 'shape'])
            
            rng = np.random.default_rng(seed=perm)
            y_train_perm = permute_labels_by_subject(subjects_train, y_train, rng=rng)
            svc.fit(X_train, y_train_perm)
            y_pred = svc.predict(X_test)
            acc = np.mean(y_pred == y_test)
            perm_df = pd.concat([perm_df, pd.DataFrame([[train_type, test_type, k, acc, perm, test_groups[k]]],
                                                        columns=['train_type', 'test_type', 'fold', 'accuracy', 'perm', 'test_subjects'])], ignore_index=True)

    perm_df.to_csv('mvpa_classification_paired_leave2out_permutation_subjshuffle.csv', index=False)

final_results = output_df.copy()
combined_results = []
train_test_pairs = final_results[['train_type', 'test_type']].drop_duplicates().reset_index(drop=True)

for _, pair in train_test_pairs.iterrows():
    train, test = pair['train_type'], pair['test_type']
    obs_mean_acc = final_results[(final_results['train_type'] == train) & (final_results['test_type'] == test)]['accuracy'].mean()
    perm_subset = perm_df[(perm_df['train_type'] == train) & (perm_df['test_type'] == test)]
    perm_means = perm_subset.groupby('perm')['accuracy'].mean().values
    p_val = np.mean(perm_means >= obs_mean_acc)
    combined_results.append({'train_type': train, 'test_type': test, 'observed_mean_accuracy': obs_mean_acc, 'p_value': p_val})

combined_results_df = pd.DataFrame(combined_results)
combined_results_df.to_csv('mvpa_combined_results.csv', index=False)