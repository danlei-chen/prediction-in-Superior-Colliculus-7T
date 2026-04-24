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

stimulation_event_files = pd.read_csv("stimulation_event_files.csv")
stimulation_img_sc = np.load("stimulation_img_sc.npy")
decision_event_files = pd.read_csv("decision_event_files.csv")
decision_img_sc = np.load("decision_img_sc.npy")
shape_event_files = pd.read_csv("shape_event_files.csv")
shape_img_sc = np.load("shape_img_sc.npy")

### leave-1-subj-out classification

def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]
NFolds = 80
subj_list = np.unique(stimulation_event_files['subject'])
rand_subj_list = shuffle(subj_list, random_state=0)
test_group_N = max(1, round(len(rand_subj_list) / NFolds))
test_groups = list(split(rand_subj_list, test_group_N))

train_test_set = [
    ('decision','stimulation'), ('decision','decision'),
    ('stimulation','decision'), ('stimulation','stimulation'),
    ('shape','stimulation'), ('shape','decision'),
    ('stimulation','shape'), ('decision','shape'), ('shape','shape')]

output_df = pd.DataFrame(columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])
svc = SVC(kernel='linear')

for k, test_subjects in enumerate(test_groups):
    for train_type, test_type in train_test_set:
        if train_type == 'stimulation':
            X_train = stimulation_img_sc[[i for i, s in enumerate(stimulation_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([stimulation_event_files['sensory_labels'][i] for i, s in enumerate(stimulation_event_files['subject']) if s not in test_subjects])
        elif train_type == 'decision':
            X_train = decision_img_sc[[i for i, s in enumerate(decision_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([decision_event_files['sensory_labels'][i] for i, s in enumerate(decision_event_files['subject']) if s not in test_subjects])
        elif train_type == 'shape':
            X_train = shape_img_sc[[i for i, s in enumerate(shape_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([shape_event_files['sensory_labels'][i] for i, s in enumerate(shape_event_files['subject']) if s not in test_subjects])
        
        if test_type == 'stimulation':
            X_test = stimulation_img_sc[[i for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([stimulation_event_files['sensory_labels'][i] for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects])
        elif test_type == 'decision':
            X_test = decision_img_sc[[i for i, s in enumerate(decision_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([decision_event_files['sensory_labels'][i] for i, s in enumerate(decision_event_files['subject']) if s in test_subjects])
        elif test_type == 'shape':
            X_test = shape_img_sc[[i for i, s in enumerate(shape_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([shape_event_files['sensory_labels'][i] for i, s in enumerate(shape_event_files['subject']) if s in test_subjects])
        
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = np.mean(y_pred == y_test)
        output_df = pd.concat([output_df, pd.DataFrame([[train_type, test_type, k, test_subjects, acc]], 
                                                        columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])], ignore_index=True)
                                                        
### paired leave-2-out classification with subject-wise label shuffling for permutation test
subj_to_label = {}
for s, lab in zip(stimulation_event_files['subject'], stimulation_event_files['sensory_labels']):
    if s not in subj_to_label:
        subj_to_label[s] = lab

# Identify the two sensory-label groups present in the stimulation data
labels_present = sorted(list(set(subj_to_label.values())))
label_a, label_b = labels_present[0], labels_present[1]

subj_list = np.unique(stimulation_event_files['subject'])
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
            X_train = stimulation_img_sc[[i for i, s in enumerate(stimulation_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([stimulation_event_files['sensory_labels'][i] for i, s in enumerate(stimulation_event_files['subject']) if s not in test_subjects])
        elif train_type == 'decision':
            X_train = decision_img_sc[[i for i, s in enumerate(decision_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([decision_event_files['sensory_labels'][i] for i, s in enumerate(decision_event_files['subject']) if s not in test_subjects])
        elif train_type == 'shape':
            X_train = shape_img_sc[[i for i, s in enumerate(shape_event_files['subject']) if s not in test_subjects], :]
            y_train = np.array([shape_event_files['sensory_labels'][i] for i, s in enumerate(shape_event_files['subject']) if s not in test_subjects])
        
        if test_type == 'stimulation':
            X_test = stimulation_img_sc[[i for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([stimulation_event_files['sensory_labels'][i] for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects])
        elif test_type == 'decision':
            X_test = decision_img_sc[[i for i, s in enumerate(decision_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([decision_event_files['sensory_labels'][i] for i, s in enumerate(decision_event_files['subject']) if s in test_subjects])
        elif test_type == 'shape':
            X_test = shape_img_sc[[i for i, s in enumerate(shape_event_files['subject']) if s in test_subjects], :]
            y_test = np.array([shape_event_files['sensory_labels'][i] for i, s in enumerate(shape_event_files['subject']) if s in test_subjects])
        
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = np.mean(y_pred == y_test)
        output_df = pd.concat([output_df, pd.DataFrame([[train_type, test_type, k, test_subjects, acc]], 
                                                        columns=['train_type', 'test_type', 'fold', 'test_subjects', 'accuracy'])], ignore_index=True)

output_df.to_csv('mvpa_classification_paired_leave2out.csv', index=False)
print(f"Mean accuracy: {np.mean(output_df['accuracy']):.4f}")

### permutation test with subject-wise label shuffling
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
                train_idx = [i for i, s in enumerate(stimulation_event_files['subject']) if s not in test_subjects]
                X_train = stimulation_img_sc[train_idx, :]
                y_train = np.array([stimulation_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([stimulation_event_files['subject'][i] for i in train_idx])
            elif train_type == 'decision':
                train_idx = [i for i, s in enumerate(decision_event_files['subject']) if s not in test_subjects]
                X_train = decision_img_sc[train_idx, :]
                y_train = np.array([decision_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([decision_event_files['subject'][i] for i in train_idx])
            elif train_type == 'shape':
                train_idx = [i for i, s in enumerate(shape_event_files['subject']) if s not in test_subjects]
                X_train = shape_img_sc[train_idx, :]
                y_train = np.array([shape_event_files['sensory_labels'][i] for i in train_idx])
                subjects_train = np.array([shape_event_files['subject'][i] for i in train_idx])
            
            if test_type == 'stimulation':
                X_test = stimulation_img_sc[[i for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects], :]
                y_test = np.array([stimulation_event_files['sensory_labels'][i] for i, s in enumerate(stimulation_event_files['subject']) if s in test_subjects])
            elif test_type == 'decision':
                X_test = decision_img_sc[[i for i, s in enumerate(decision_event_files['subject']) if s in test_subjects], :]
                y_test = np.array([decision_event_files['sensory_labels'][i] for i, s in enumerate(decision_event_files['subject']) if s in test_subjects])
            elif test_type == 'shape':
                X_test = shape_img_sc[[i for i, s in enumerate(shape_event_files['subject']) if s in test_subjects], :]
                y_test = np.array([shape_event_files['sensory_labels'][i] for i, s in enumerate(shape_event_files['subject']) if s in test_subjects])
            
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
