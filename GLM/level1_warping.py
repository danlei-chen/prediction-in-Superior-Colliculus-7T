import os
import glob
import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import ball
import nipype.pipeline.engine as pe
from nipype import IdentityInterface
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility.wrappers import Function
from nipype.interfaces.spm.preprocess import DARTEL, CreateWarped

def files_from_template(identity_list, template):
    if type(identity_list) != list:
        identity_list = [identity_list]
    out_list = []
    for x in glob.glob(template):
        if any(subj in x for subj in identity_list):
            out_list.append(x)
    return out_list

def clust_thresh(img, thresh=95, cluster_k=50):
    out_labeled = np.empty((img.shape[0], img.shape[1], img.shape[2]))
    data = img[np.where(~np.isnan(img))]
    img = np.nan_to_num(img)
    img[img < np.percentile(data, thresh)] = 0
    label_map, n_labels = label(img)
    lab_val = 1
    if type(cluster_k) == int:
        for label_ in range(1, n_labels + 1):
            if np.sum(label_map == label_) >= cluster_k:
                out_labeled[label_map == label_] = lab_val
                lab_val = lab_val + 1
    else:
        for k in cluster_k:
            for label_ in range(1, n_labels + 1):
                if np.sum(label_map == label_) >= k:
                    out_labeled[label_map == label_] = lab_val
                    lab_val = lab_val + 1
            if lab_val > 1:
                break
    return out_labeled

def mask_img(img_file, mask_file, out_format='file', work_dir=''):
    mask = nib.load(mask_file)
    mask_name = '_' + mask_file.split('/')[-1].split('.')[0]
    img_name = img_file.split('/')[-1].split('.')[0]
    img = nib.load(img_file)
    data = nib.load(img_file).get_fdata()
    if mask.shape != data.shape[0:3]:
        new_shape = list(data.shape[0:3])
        while len(new_shape) != len(mask.shape):
            new_shape.append(1)
        mask_data = resize(mask.get_fdata(), new_shape, order=0, preserve_range=True)
    else:
        mask_data = mask.get_fdata()
    data[mask_data != 1] = 0
    if out_format == 'file':
        if work_dir == '':
            work_dir = os.getcwd()
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_path = os.path.join(work_dir, img_name + mask_name + '.nii.gz')
        nib.save(out_img, out_path)
        return out_path
    else:
        return data

def create_roi_template(subj_list, p_thresh_list, template, work_dir, out_dir, space_mask):
    import pandas as pd
    for subj in subj_list:
        try:
            img_file = files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_cluster_map.nii.gz'))[0]
        except:
            img_file = files_from_template(subj, template)[0]
            img_info = nib.load(img_file)
            img = mask_img(img_file, space_mask, out_format='array')
            all_labeled = None
            for thresh in p_thresh_list:
                img_labeled = clust_thresh(img, cluster_k=[50], thresh=thresh)
                if all_labeled is None:
                    all_labeled = img_labeled[..., np.newaxis]
                else:
                    all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3)
            roi_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
            roi_img.header['cal_max'] = np.max(all_labeled)
            roi_img.header['cal_min'] = 0
            for d in [os.path.join(work_dir, 'subj_clusts'), os.path.join(out_dir, 'subj_clusts')]:
                os.makedirs(d, exist_ok=True)
                nib.save(roi_img, os.path.join(d, subj + '_cluster_map.nii.gz'))
    
    all_subj_data = None
    for subj in subj_list:
        img_file = files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_cluster_map.nii.gz'))[0]
        img = nib.load(img_file).get_fdata()
        if all_subj_data is None:
            all_subj_data = img[..., np.newaxis]
        else:
            all_subj_data = np.append(all_subj_data, img[..., np.newaxis], axis=3)
    
    roi_template = np.copy(all_subj_data[..., 0, :])
    roi_template[roi_template != 1] = 0
    roi_template = np.mean(roi_template, axis=3)
    roi_report = pd.DataFrame(columns=['sub', 'thresh', 'clust', 'corr', 'iter', 'FLAG'])
    roi_report['sub'] = subj_list
    roi_report['iter'] = 0
    roi_report['FLAG'] = ''
    roi_report = roi_report.set_index('sub')
    
    converged = False
    while not converged:
        roi_report['iter'] = roi_report['iter'] + 1
        new_template = np.zeros(list(all_subj_data.shape[0:3]) + [all_subj_data.shape[-1]])
        for subj_idx, subj in enumerate(subj_list):
            corr_val = -101
            for thresh_idx, thresh in enumerate(p_thresh_list):
                if np.max(all_subj_data[..., thresh_idx, subj_idx]) > 0:
                    for cluster in np.unique(all_subj_data[..., thresh_idx, subj_idx]):
                        if cluster == 0:
                            continue
                        test_array = np.copy(all_subj_data[..., thresh_idx, subj_idx])
                        test_array[test_array != cluster] = 0
                        test_array[test_array == cluster] = 1
                        clust_corr = np.corrcoef(np.ndarray.flatten(roi_template),
                                               np.ndarray.flatten(test_array))[0, 1]
                        if clust_corr > corr_val:
                            roi_report.at[subj, 'thresh'] = thresh
                            roi_report.at[subj, 'clust'] = cluster
                            roi_report.at[subj, 'corr'] = clust_corr
                            if clust_corr < 0.3:
                                roi_report.at[subj, 'FLAG'] = 'CHECK'
                            else:
                                roi_report.at[subj, 'FLAG'] = ''
                            new_template[..., subj_idx] = test_array
                            corr_val = clust_corr
        
        if np.array_equal(np.around(roi_template, 4), np.around(np.mean(new_template, axis=3), 4)):
            converged = True
        else:
            roi_template = np.mean(new_template, axis=3)
    
    for img_idx, subj in enumerate(subj_list):
        img_info = nib.load(files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_cluster_map.nii.gz'))[0])
        subj_temp = nib.Nifti1Image(new_template[..., img_idx], img_info.affine, img_info.header)
        subj_temp.header['cal_max'] = 1
        subj_temp.header['cal_min'] = 0
        for d in [os.path.join(work_dir, 'templates'), os.path.join(out_dir, 'templates')]:
            os.makedirs(d, exist_ok=True)
            nib.save(subj_temp, os.path.join(d, subj + '_roi_template.nii.gz'))
    
    img_info = nib.load(files_from_template(subj_list[0], os.path.join(work_dir, 'subj_clusts', '*_cluster_map.nii.gz'))[0])
    roi_temp_img = nib.Nifti1Image(roi_template, img_info.affine, img_info.header)
    roi_temp_img.header['cal_max'] = 1
    roi_temp_img.header['cal_min'] = 0
    for d in [os.path.join(work_dir, 'templates'), os.path.join(out_dir, 'templates')]:
        os.makedirs(d, exist_ok=True)
        nib.save(roi_temp_img, os.path.join(d, 'MEAN_roi_template.nii.gz'))
    
    roi_report.to_csv(os.path.join(work_dir, 'templates', 'roi_report.csv'))
    roi_report.to_csv(os.path.join(out_dir, 'templates', 'roi_report.csv'))

def make_roi_masks(subj_list, data_template, gm_template, work_dir, out_dir, gm_thresh=0.5, gm_spline=3, dilation_r=2, x_minmax=False, y_minmax=False, z_minmax=False):
    for subj in subj_list:
        img_file = nib.load(files_from_template(subj, data_template)[0])
        img = img_file.get_fdata()
        gm_file = nib.load(files_from_template(subj, gm_template)[0])
        gm_img = gm_file.get_fdata()
        if img.shape[0:3] != gm_img.shape[0:3]:
            gm_img = resize(gm_img, img.shape[0:3], order=gm_spline, preserve_range=True)
        gm_img[np.where(gm_img < gm_thresh)] = 0
        if not x_minmax:
            x_minmax = [0, list(img.shape)[0]]
        if not y_minmax:
            y_minmax = [0, list(img.shape)[1]]
        if not z_minmax:
            z_minmax = [0, list(img.shape)[2]]
        loc_mask = np.zeros(list(img.shape))
        loc_mask[x_minmax[0]:x_minmax[1], y_minmax[0]:y_minmax[1], z_minmax[0]:z_minmax[1]] = 1
        roi = binary_dilation(img, ball(dilation_r)).astype(img.dtype) - img
        roi = roi * gm_img * loc_mask
        roi_file = nib.Nifti1Image(roi, img_file.affine, img_file.header)
        for d in [os.path.join(work_dir, 'roi_mask'), os.path.join(out_dir, 'roi_mask')]:
            os.makedirs(d, exist_ok=True)
            nib.save(roi_file, os.path.join(d, subj + '_roi_mask.nii.gz'))

def create_dartel_wf(subj_list, file_template, work_dir, out_dir):
    dartel_wf = pe.Workflow(name='DARTEL_wf')
    dartel_wf.base_dir = work_dir
    images = files_from_template(subj_list, file_template)
    dartel = pe.Node(interface=DARTEL(), name='dartel')
    dartel.inputs.image_files = [images]
    dartel_warp = pe.Node(interface=CreateWarped(), name='dartel_warp')
    dartel_warp.inputs.image_files = images
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')
    sinker.inputs.base_directory = out_dir
    dartel_wf.connect([(dartel, dartel_warp, [('dartel_flow_fields', 'flowfield_files')]),
                       (dartel, sinker, [('final_template_file', 'avg_template'),
                                        ('template_files', 'avg_template.@template_stages'),
                                        ('dartel_flow_fields', 'dartel_flow')]),
                       (dartel_warp, sinker, [('warped_files', 'warped_roi')])])
    return dartel_wf

def setup_dartel_warp_wf(subj_list, data_template, warp_template, work_dir, out_dir):
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    apply_warp_wf = pe.Workflow(name='apply_warp_wf')
    apply_warp_wf.base_dir = work_dir
    inputspec = pe.Node(IdentityInterface(fields=['file_list', 'warp_list']), name='inputspec')
    inputspec.inputs.file_list = sorted(files_from_template(subj_list, data_template))
    inputspec.inputs.warp_list = sorted(files_from_template(subj_list, warp_template))
    
    def rename_list(in_list):
        out_list = []
        for file in in_list:
            file_in = nib.load(file)
            out_file = os.path.join(os.getcwd(), '_'.join(file.split('/')[-3:]))
            nib.save(file_in, out_file)
            out_list.append(out_file)
        return out_list
    
    rename = pe.Node(Function(input_names=['in_list'], output_names=['out_list'], function=rename_list), name='rename')
    warp_data = pe.Node(interface=CreateWarped(), name='warp_data')
    sinker = pe.Node(DataSink(), name='sinker')
    sinker.inputs.base_directory = out_dir
    apply_warp_wf.connect([(inputspec, rename, [('file_list', 'in_list')]),
                           (inputspec, warp_data, [('warp_list', 'flowfield_files')])])
    
    if any('nii.gz' in file for file in files_from_template(subj_list, data_template)):
        from nipype.algorithms.misc import Gunzip
        gunzip = pe.MapNode(interface=Gunzip(), name='gunzip', iterfield=['in_file'])
        apply_warp_wf.connect([(rename, gunzip, [('out_list', 'in_file')]),
                               (gunzip, warp_data, [('out_file', 'image_files')]),
                               (warp_data, sinker, [('warped_files', 'warped_files')])])
    else:
        apply_warp_wf.connect([(rename, warp_data, [('out_list', 'image_files')]),
                               (warp_data, sinker, [('warped_files', 'warped_files')])])
    return apply_warp_wf

subj_list = ['sub-001', 'sub-002', 'sub-003']
work_dir = '/work/dartel'
data_dir = '/data/bids'
out_dir = '/output/dartel'

roi_template = os.path.join(work_dir, 'roi_mask', 'ROI*subject-*.nii')
roi_dartel = create_dartel_wf(subj_list, roi_template, work_dir, out_dir)
roi_dartel.run(plugin='MultiProc')

for subj_run in subj_list:
    subj = subj_run.split('_')[0] if '_' in subj_run else subj_run
    run = subj_run.split('_')[1] if '_' in subj_run else '01'
    image_list = glob.glob(f'/data/level1_results/project_a/mni/{subj}_task-*{run}/model/_*/_modelestimate0/cope*.nii.gz')
    image_list = sorted([i for i in image_list if i.split('/')[-1] not in [j.split('_')[-1].split('.nii.gz')[0] for j in glob.glob(f'/output/w_{subj}*{run}*.nii.gz')]])
    
    data_templates = {}
    for image_file in image_list:
        key = subj + '_' + run + '_' + image_file.split('/')[-1].split('.nii')[0]
        data_templates[key] = image_file
    
    warp_template = f'/work/flowfields/u_ROI*{subj}_{run}*_Template_resampled.nii'
    for data_key in data_templates:
        dartel_warp = setup_dartel_warp_wf(subj, data_templates[data_key], warp_template, 
                                          os.path.join(work_dir, data_key), os.path.join(out_dir, data_key))
        dartel_warp.run(plugin='MultiProc')