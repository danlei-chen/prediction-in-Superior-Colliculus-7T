import numpy as np
import pandas as pd
import os, sys
import nibabel as nib
import glob
import math

# subj_list = [i.split('/')[-1] for i in glob.glob('/Users/chendanlei/Desktop/U01/level1_files/wm_3vs1/SC_warp/*')]
# subj_list.sort()
subj_list = ['sub-012', 'sub-013', 'sub-015', 'sub-019', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-030', 'sub-032', 'sub-034', 'sub-037', 'sub-039', 'sub-042b', 'sub-043', 'sub-044', 'sub-045', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-052', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-057', 'sub-058', 'sub-059', 'sub-060', 'sub-061', 'sub-062', 'sub-063', 'sub-064', 'sub-065', 'sub-066', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 'sub-072', 'sub-073', 'sub-074', 'sub-078', 'sub-080', 'sub-081', 'sub-082', 'sub-083', 'sub-084', 'sub-085', 'sub-086', 'sub-087', 'sub-088', 'sub-090', 'sub-091', 'sub-092', 'sub-093', 'sub-094', 'sub-095', 'sub-098', 'sub-099', 'sub-100', 'sub-101', 'sub-102', 'sub-104', 'sub-105', 'sub-106', 'sub-110', 'sub-111', 'sub-112', 'sub-114', 'sub-117', 'sub-118', 'sub-119', 'sub-120b', 'sub-122b', 'sub-124', 'sub-127', 'sub-128', 'sub-131', 'sub-133', 'sub-134', 'sub-135', 'sub-137', 'sub-138', 'sub-139b']
files = glob.glob('/Volumes/GoogleDrive/My Drive/U01/working_memory/data/event_files/')

for subj in subj_list:
    
    subj_df = pd.read_csv(glob.glob('/Volumes/GoogleDrive/My Drive/U01/working_memory/data/event_files/'+subj+'*.tsv')[0], sep='\t', header=0)



