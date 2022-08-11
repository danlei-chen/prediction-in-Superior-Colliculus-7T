# ################################################################################################################################################
# ################################################################################################################################################
# #################################################################################################################################################1_DARTEL_mask_PAG
# # from jtnipyutil.fsmap import create_aqueduct_template, make_PAG_masks, create_DARTEL_wf
# def files_from_template(identity_list, template):
#	 '''
#	 Uses glob to grab all matches to a template, then subsets the list with identifier.
#	 Input [Mandatory]:
#		 identity_list: string or list of strings, to grab subset of glob search.
#			 e.g. 'sub-01', or ['sub-01', 'sub-02', 'sub-03']
#		 template: string denoting a path, with wildcards, to be used in glob.
#			 e.g. '/home/neuro/data/smoothsub-*/model/sub-*.nii.gz'
#	 Output:
#		 out_list: list of file paths, first from the glob template,
#		 then subseted by identifier.
#	 '''
#	 import glob
#	 out_list = []
#	 if type(identity_list) != list:
#		 assert (type(identity_list) == str), 'identifier must be either a string, or a list of string'
#		 identity_list = [identity_list]
#	 for x in glob.glob(template):
#		 if any(subj in x for subj in identity_list):
#			 out_list.append(x)
#	 return out_list

# def clust_thresh(img, thresh=95, cluster_k=50):
#	 '''
#	 Thresholds an array, then computes sptial cluster.
#	 Input [Mandatory]:
#		 img: 3d array - e.g. from nib.get_data()
#	 Input:
#		 thresh: % threshold extent.
#			 Default = 95
#		 cluster_k: k-voxel cluster extent. integer or list.
#			 Default = 50.
#			 A list can be used to give fallback clusters. e.g. cluster_k= [50, 40]
#				 In this case, the first threhsold is used,
#				 and if nothing passes it then we move onto the next.
#	 Output:
#		 out_labeled: 3d array, with values 1:N for clusters, and 0 otherwise.
#	 '''
#	 import nibabel as nib
#	 import numpy as np
#	 from scipy.ndimage import label
#	 out_labeled = np.empty((img.shape[0], img.shape[1],img.shape[2]))
#	 data = img[np.where(~np.isnan(img))] # strip out data, to avoid np.nanpercentile.
#	 img = np.nan_to_num(img)
#	 img[img < np.percentile(data, thresh)] = 0 #threshold residuals.
#	 label_map, n_labels = label(img) # label remaining voxels.
#	 lab_val = 1 # this is so that labels are ordered sequentially, rather than having gaps.
#	 if type(cluster_k) == int:
#		 print(('looking for clusters > %s voxels, at %s%% threshold') % (cluster_k, thresh))
#		 for label_ in range(1, n_labels+1): # addition is to match labels, which are base 1.
#			 if np.sum(label_map==label_) >= cluster_k:
#				 out_labeled[label_map==label_] = lab_val # zero any clusters below cluster threshold.
#				 print(('saving cluster %s') % lab_val)
#				 lab_val = lab_val+1 # add to counter.
#	 else:
#		 assert (type(cluster_k) == list), 'cluster_k must either be an integer or list.'
#		 for k in cluster_k:
#			 print(('looking for clusters > %s voxels') % k)
#			 for label_ in range(1, n_labels+1):
#				 if np.sum(label_map==label_) >= k:
#					 out_labeled[label_map==label_] = lab_val
#					 print(('saving cluster %s at min %s voxels') % (lab_val, k))
#					 lab_val = lab_val+1
#			 if lab_val > 1: # if we find any clusters above the threshold, then move on. Otherwise, try another threshold.
#				 break

#	 return out_labeled

# def mask_img(img_file, mask_file, work_dir = '', out_format = 'file', inclu_exclu = 'inclusive', spline = 0):
#	 '''
#	 Applies a mask, converting to reference space if necessary, using nearest neighbor classification.
#	 Input [Mandatory]:
#		 img_file: path to a nifti file to be masked. Can be 3d or 4d.
#		 mask_file: path to a nifti mask file. Does not need to match dimensions of img_file
#		 work_dir: [default = ''] path to directory to save masked file. Required if out_format = 'file'.
#		 out_format: [default = 'file'] Options are 'file', or 'array'.
#		 inclu_exclu: [default = 'exclusive'] Options are 'exclusive' and 'inclusive'
#	 Output
#		 out_img: Either a nifti file, or a np array, depending on out_format.
#	 '''
#	 import numpy as np
#	 import nibabel as nib
#	 import os.path
#	 from skimage.transform import resize
#	 print(('loading mask file: %s') % mask_file)
#	 mask = nib.load(mask_file)
#	 mask_name = '_'+mask_file.split('/')[-1].split('.')[0]
#	 img_name = img_file.split('/')[-1].split('.')[0]
#	 print(('loading img file: %s') % img_file)
#	 img = nib.load(img_file) # grab data
#	 data = nib.load(img_file).get_data() # grab data
#	 if mask.shape != data.shape[0:3]:
#		 new_shape = list(data.shape[0:3])
#		 while len(new_shape) != len(mask.shape): # add extra dimensions, in case ref img is 4d.
#			 new_shape.append(1)
#		 mask = resize(mask.get_data(), new_shape, order = spline, preserve_range=True) # interpolate mask to native space.
#	 else:
#		 mask = mask.get_data()
#	 if inclu_exclu == 'inclusive':
#		 data[mask!=1] = 0 # mask
#	 else:
#		 assert(inclu_exclu == 'exclusive'), 'mask must be either inclusive or exclusive'
#		 data[mask==1] = 0 # mask

#	 if out_format == 'file':
#		 if work_dir == '':
#			 print('No save directory specified, saving to current working direction')
#			 work_dir = os.getcwd()
#		 out_img = nib.Nifti1Image(data, img.affine, img.header)
#		 nib.save(out_img, os.path.join(work_dir, img_name + mask_name + '.nii.gz'))
#		 out_img = os.path.join(work_dir, img_name + mask_name + '.nii.gz')
#	 else:
#		 assert (out_format == 'array'), 'out_format is neither file, or array.'
#		 out_img = data

#	 return out_img

# def create_aqueduct_template(subj_list, p_thresh_list, template, work_dir, out_dir, space_mask):
#	 '''
#	 The workflow takes the following as input to wf.inputs.inputspec
#	 Input [Mandatory]:
#		 subj_list: list of subject IDs
#			 e.g. [sub-001, sub-002]
#		 p_thresh_list: list of floats representing p thresholds. Applied to resdiduals.
#			 e.g. [95, 97.5, 99.9]
#		 template: string to identify all PAG aqueduct files (using glob).
#			 e.g. template = '/home/neuro/func/sub-001/sigmasquareds.nii.gz'
#				 The template can identify a larger set f files, and the subject_list will grab a subset.
#					 e.g. The template may grab sub-001, sub-002, sub-003 ...
#					 But if the subject_list only includes sub-001, then only sub-001 will be used.
#					 This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)
#		 work_dir: string, denoting path to working directory.
#		 out_dir: string, denoting output directory (results saved to work directory and output)
#		 space_mask: string, denoting path to PAG search region mask.
#	 Output:
#		 /subj_cluster/sub-xxx_sigmasquare_clusts.nii.gz
#			 folder within work_dir, listing nii.gz images with all clusters t the specified p thresholds and cluster extents (default = 50 only)
#		 /templates/sub-xxx_aqueduct_template.nii.gz
#			 all subject aqueduct templates output as nii.gz files
#		 /templates/MEAN_aqueduct_template.nii.gz
#			 average of all aqueduct templates, which was used to help converge on the correct cluster.
#		 /templates/report.csv
#			 report on output, listing subject, threshold used, cluster, corelation with the average, and how many iterations it took to settle on an answer. Results where corr < .3 are flagged.
#	 '''
#	 import nibabel as nib
#	 import numpy as np
#	 import pandas as pd
#	 import os
#	 # from jtnipyutil.util import files_from_template, clust_thresh, mask_img

#	 for subj in subj_list: # For each subject, create aqueduct template file wtih all thresholded clusters.
#		 print('creating aqueduct template for %s' % subj)
#		 try:
#			 img_info = nib.load(files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
#		 except:
#			 img_file  = files_from_template(subj, template)[0]
#			 img_info = nib.load(img_file)
#			 img = mask_img(img_file, space_mask, out_format = 'array') # loading done here. Slow.
#			 # img = np.nanmean(img, axis=3) # Average data along time.
#			 for thresh in p_thresh_list:
#				 img_labeled = clust_thresh(img, cluster_k=[50], thresh = thresh)
#				 if thresh == p_thresh_list[0]:
#					 all_labeled = img_labeled[..., np.newaxis]
#				 else:
#					 all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3) # stack thresholds along 4th dim.
#			 pag_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
#			 pag_img.header['cal_max'] = np.max(all_labeled) # fix header info
#			 pag_img.header['cal_min'] = 0 # fix header info
#			 try:
#				 nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
#				 nib.save(pag_img, os.path.join(out_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
#			 except:
#				 os.makedirs(os.path.join(work_dir, 'subj_clusts'))
#				 os.makedirs(os.path.join(out_dir, 'subj_clusts'))
#				 nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
#				 nib.save(pag_img, os.path.join(out_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))

#	 ## gather all subjects clusters/thresholds into a 5d array. ##########################################
#	 for subj in subj_list:
#		 img_file = files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))
#		 print(('getting data from %s') % img_file[0])
#		 img = nib.load(img_file[0]).get_data()
#		 if subj == subj_list[0]:
#			 all_subj_data = img[..., np.newaxis]
#		 else:
#			 all_subj_data = np.append(all_subj_data, img[...,np.newaxis], axis=4)

#	 ## get mean across defaults: threshold (95) and cluster (1) ##########################################
#	 # This establishes a template to judge which threshold fits it best.
#	 # Average is across all subjects.
#	 aq_template = np.copy(all_subj_data[...,0,:])
#	 aq_template[aq_template != 1] = 0
#	 aq_template = np.mean(aq_template, axis=3)
#	 # set up report.
#	 aq_report = pd.DataFrame(columns=['sub', 'thresh', 'clust', 'corr', 'iter', 'FLAG'], data={'sub':subj_list, 'iter':[0]*len(subj_list), 'FLAG':['']*len(subj_list)})
#	 aq_report = aq_report.set_index('sub')
#	 while True:
#		 aq_report['iter'] = aq_report['iter'] + 1
#		 new_template = np.zeros(list(all_subj_data.shape[0:3]) + [all_subj_data.shape[-1]])
#		 for subj_idx, subj in enumerate(subj_list):
#			 corr_val = -101
#			 for thresh_idx, thresh in enumerate(p_thresh_list):
#				 if np.max(all_subj_data[..., thresh_idx, subj_idx]) > 0:
#					 for cluster in np.unique(all_subj_data[...,thresh_idx, subj_idx]):
#						 if cluster == 0:
#							 continue
#						 else:
#							 print(('checking sub %s, thresh %s, clust %s') % (subj, thresh, cluster))
#							 test_array = np.copy(all_subj_data[...,thresh_idx, subj_idx]) # binarize array being tested.
#							 test_array[test_array != cluster] = 0
#							 test_array[test_array == cluster] = 1
#							 clust_corr = np.corrcoef(np.ndarray.flatten(aq_template), # correlate with group mean.
#													   np.ndarray.flatten(test_array))[0,1]
#							 if clust_corr > corr_val:
#								 print(('sub %s, thresh %s, clust %s, corr =  %s (prev max corr = %s)') %
#									   (subj, thresh, cluster, clust_corr, corr_val))
#								 aq_report.at[subj, 'thresh'] = thresh
#								 aq_report.at[subj, 'clust'] = cluster
#								 aq_report.at[subj, 'corr'] = clust_corr
#								 if clust_corr < .3:
#									 aq_report.at[subj, 'FLAG'] = 'CHECK'
#								 else:
#									 aq_report.at[subj, 'FLAG'] = ''
#								 new_template[...,subj_idx] = test_array
#								 corr_val = clust_corr

#		 if np.array_equal(np.around(aq_template, 4), np.around(np.mean(new_template, axis=3), 4)):
#			 print('We have converged on a stable average for aq_template.')
#			 break
#		 else:
#			 aq_template = np.mean(new_template, axis=3)
#			 print('new aq_template differs from previous iteration. Performing another iteration.')

#	 for img_idx in range(0, new_template.shape[-1]):
#		 print(('Saving aqueduct for %s') % (subj_list[img_idx]))
#		 img_info = nib.load(files_from_template(subj_list[img_idx], os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
#		 subj_temp = nib.Nifti1Image(new_template[...,img_idx], img_info.affine, img_info.header)
#		 subj_temp.header['cal_max'] = 1 # fix header info
#		 subj_temp.header['cal_min'] = 0 # fix header info
#		 try:
#			 nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
#			 nib.save(subj_temp, os.path.join(out_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
#		 except:
#			 os.makedirs(os.path.join(work_dir, 'templates'))
#			 os.makedirs(os.path.join(out_dir, 'templates'))
#			 nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
#			 nib.save(subj_temp, os.path.join(out_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))

#	 print('Saving aqueduct mean template.')
#	 img_info = nib.load(files_from_template(subj_list[0], os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
#	 aq_temp_img = nib.Nifti1Image(aq_template, img_info.affine, img_info.header)
#	 aq_temp_img.header['cal_max'] = 1 # fix header info
#	 aq_temp_img.header['cal_min'] = 0 # fix header info
#	 nib.save(aq_temp_img, os.path.join(work_dir, 'templates', 'MEAN_aqueduct_template.nii.gz'))
#	 nib.save(aq_temp_img, os.path.join(out_dir, 'templates', 'MEAN_aqueduct_template.nii.gz'))

#	 print('Saving report')
#	 aq_report.to_csv(os.path.join(work_dir, 'templates', 'report.csv'))
#	 aq_report.to_csv(os.path.join(out_dir, 'templates', 'report.csv'))

# def make_PAG_masks(subj_list, data_template, gm_template, work_dir, out_dir, gm_thresh = .5, gm_spline=3, dilation_r=2, x_minmax=False, y_minmax=False, z_minmax=False):
#	 '''
#	 subj_list: list of subject IDs
#		 e.g. [sub-001, sub-002]
#	 data_template: string to identify all PAG aqueduct files (using glob).
#		 e.g. data_template = os.path.join(work_dir, 'templates', '*_aqueduct_template.nii.gz')
#			 The template can identify a larger set of files, and the subject_list will grab a subset.
#				 e.g. The template may grab sub-001, sub-002, sub-003 ...
#				 But if the subject_list only includes sub-001, then only sub-001 will be used.
#				 This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)
#	 gm_template: string to identify all PAG aqueduct files (using glob).
#		 e.g. gm_template = os.path.join('work_dir, 'gm', '*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')
#	 work_dir: string, denoting path to working directory.
#	 out_dir: string, denoting output directory (results saved to work directory and output)
#	 gm_thresh: float specifying the probability to threshold gray matter mask.
#		 Default: .5
#	 gm_spline: integer specifying the spline order to use to reslice gray matter to native space (if necessary)
#		 Default: 3
#	 dilation_r: integer specifying the number of voxels to dilate the aqueduct.
#		 Default: 2
#	 x_minmax: list of 2 integers, denoting min and max X voxels to include in mask.
#		 e.g. [0,176]. Defaults to full range of PAG aqueduct image.
#	 y_minmax: list of 2 integers, denoting min and max Y voxels to include in mask.
#			 e.g. [82,100]. Defaults to full range of PAG aqueduct image.
#	 z_minmax: list of 2 integers, denoting min and max Z voxels to include in mask.
#		 e.g. [58,176]. Defaults to full range of PAG aqueduct image.
#	 '''
#	 import nibabel as nib
#	 import numpy as np
#	 from skimage.transform import resize
#	 from scipy.ndimage.morphology import binary_dilation
#	 from skimage.morphology import ball
#	 import os
#	 # from jtnipyutil.util import files_from_template, clust_thresh, mask_img

#	 for subj in subj_list:
#		 print('making PAG mask for subject: %s' % subj)
#		 # get aqueduct.
#		 img_file = nib.load(files_from_template(subj, data_template)[0])
#		 img = img_file.get_data()
#		 # get gray matter, binarize at threshold.
#		 gm_file = nib.load(files_from_template(subj, gm_template)[0])
#		 gm_img = gm_file.get_data()
#		 if img.shape[0:3] !=  gm_img.shape[0:3]:
#			 gm_img = resize(gm_img, img.shape[0:3], order=gm_spline, preserve_range=True)
#		 gm_img[np.where(gm_img < gm_thresh)] = 0
#		 # create mask for PAG location.
#		 if not x_minmax:
#			 x_minmax = [0,list(img.shape)[0]]
#		 if not y_minmax:
#			 y_minmax = [0,list(img.shape)[1]]
#		 if not z_minmax:
#			 z_minmax = [0,list(img.shape)[2]]
#		 loc_mask = np.zeros(list(img.shape))
#		 loc_mask[x_minmax[0]:x_minmax[1],
#			  y_minmax[0]:y_minmax[1],
#			  z_minmax[0]:z_minmax[1]] = 1
#		 # dilate and subtract original aqueduct.
#		 pag = binary_dilation(img, ball(dilation_r)).astype(img.dtype) - img
#		 pag = pag*gm_img # multiply by thresholded gm probability mask.
#		 pag = pag*loc_mask # threshold by general PAG location cutoffs.
#		 pag_file = nib.Nifti1Image(pag, img_file.affine, img_file.header)
#		 try:
#			 nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii'))
#			 nib.save(pag_file, os.path.join(out_dir, 'pag_mask', subj+'_pag_mask.nii'))
#		 except:
#			 os.makedirs(os.path.join(work_dir, 'pag_mask'))
#			 os.makedirs(os.path.join(out_dir, 'pag_mask'))
#			 nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii'))
#			 nib.save(pag_file, os.path.join(out_dir, 'pag_mask', subj+'_pag_mask.nii'))

# def create_DARTEL_wf(subj_list, file_template, work_dir, out_dir):
#	 '''
#	 Aligns all images to a template (average of all images), then warps images into MNI space (using an SPM tissue probability map, see https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf, section 25.4).
#	 subj_list: list of subject IDs
#		 e.g. [sub-001, sub-002]
#	 file_template: string to identify all files to align (using glob).
#		 e.g. file_template = os.path.join(work_dir, 'pag_mask', '*_pag_mask.nii')
#			 The template can identify a larger set of files, and the subject_list will grab a subset.
#				 e.g. The template may grab sub-001, sub-002, sub-003 ...
#				 But if the subject_list only includes sub-001, then only sub-001 will be used.
#				 This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)
#	 work_dir: string, denoting path to working directory.
#	 out_dir: string, denoting output directory (results saved to work directory and output)
#	 '''
#	 import nibabel as nib
#	 import numpy as np
#	 from nipype.interfaces.spm.preprocess import DARTEL, CreateWarped
#	 from nipype.interfaces.io import DataSink
#	 import nipype.pipeline.engine as pe
#	 import os
#	 # from jtnipyutil.util import files_from_template
#	 # set up workflow.
#	 DARTEL_wf = pe.Workflow(name='DARTEL_wf')
#	 DARTEL_wf.base_dir = work_dir

#	 # get images
#	 images = files_from_template(subj_list, file_template)

#	 # set up DARTEL.
#	 dartel = pe.Node(interface=DARTEL(), name='dartel')
#	 dartel.inputs.image_files = [images]

#	 dartel_warp = pe.Node(interface=CreateWarped(), name='dartel_warp')
#	 dartel_warp.inputs.image_files = images
#	 #	 warp_data.inputs.flowfield_files = # from inputspec

#	 ################## Setup datasink.
#	 sinker = pe.Node(DataSink(parameterization=True), name='sinker')
#	 sinker.inputs.base_directory = out_dir

#	 DARTEL_wf.connect([(dartel, dartel_warp, [('dartel_flow_fields', 'flowfield_files')]),
#						(dartel, sinker, [('final_template_file', 'avg_template'),
#										 ('template_files', 'avg_template.@template_stages'),
#										 ('dartel_flow_fields', 'dartel_flow')]),
#						(dartel_warp, sinker, [('warped_files', 'warped_PAG')])])

#	 return DARTEL_wf

# import os

# subj_list = os.environ.get("SUBJ_LIST").split(" ")
# subj_list = ['sub-014_run-01', 'sub-014_run-02', 'sub-014_run-03', 'sub-014_run-04', 'sub-014_run-05']
# print(subj_list)

# # p_thresh_list = [99.99, 99.995, 99.999]
# # variance_template = '/scratch/data/sub-*_run-*/sigmasquareds.nii.gz'
# work_dir = '/scratch/wrkdir'
# data_dir = '/scratch/data'
# out_dir = '/scratch/output'
# # space_mask =  '/scratch/wrkdir/search_region.nii'
# # create_aqueduct_template(subj_list, p_thresh_list, variance_template, work_dir, out_dir, space_mask)

# # aqueduct_template = os.path.join(work_dir, 'templates', '*_aqueduct_template.nii.gz')
# # gm_template = os.path.join(data_dir, 'sub-*_run-*', '*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')
# # y_minmax = [82,100]
# # z_minmax = [58,176]
# # make_PAG_masks(subj_list, aqueduct_template, gm_template, work_dir, out_dir, y_minmax=y_minmax, z_minmax=z_minmax)

# # PAG_template = os.path.join(work_dir, 'pag_mask', '*_pag_mask.nii')
# # PAG_DARTEL = create_DARTEL_wf(subj_list, PAG_template, work_dir, out_dir)
# # PAG_DARTEL.run(plugin='MultiProc')
# SC_template = os.path.join(work_dir, 'SC_mask', 'SC*sub-*.nii')
# SC_DARTEL = create_DARTEL_wf(subj_list, SC_template, work_dir, out_dir)
# SC_DARTEL.run(plugin='MultiProc')

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
#2_warp_to_PAG
# from jtnipyutil.fsmap import setup_DARTEL_warp_wf
def setup_DARTEL_warp_wf(subj_list, data_template, warp_template, work_dir, out_dir):
	'''
	subj_list: list of strings for each subject
		e.g. ['sub-001', 'sub-002', 'sub-003']
	data_template: string to identify all data files (using glob).
			e.g. template = '/home/neuro/data/rest1_AROMA/nosmooth/sub-*/model/sub-*/_modelestimate0/res4d.nii.gz'
				The template can identify a larger set of files, and the subject_list will grab a subset.
					e.g. The template may grab sub-001, sub-002, sub-003 ...
					But if the subject_list only includes sub-001, then only sub-001 will be used.
					This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)
	warp_template: string to identify all dartel flowfield files (using glob).
		same as above.
		Dartel flowfield files are made by create_DARTEL_wf,
			also see jtnipyutil.fsmap.make_PAG_masks, and jtnipyutil.fsmap.create_aqueduct_template
	work_dir: string naming directory to store work.
	out_dir: string naming directory for output.
	'''
	import os
	import nibabel as nib
	import numpy as np
	import nipype.pipeline.engine as pe
	from nipype import IdentityInterface
	from nipype.interfaces.io import DataSink
	from nipype.interfaces.utility.wrappers import Function
	from nipype.interfaces.spm.preprocess import CreateWarped
	# from jtnipyutil.util import files_from_template
	# create working directory if necessary.
	if not os.path.isdir(work_dir):
		os.makedirs(work_dir)
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	# set up data warp workflow
	apply_warp_wf = pe.Workflow(name='apply_warp_wf')
	apply_warp_wf.base_dir = work_dir
	# set up file lists
	inputspec = pe.Node(IdentityInterface(
		fields=['file_list',
				'warp_list']),
					   name='inputspec')
	inputspec.inputs.file_list = files_from_template(subj_list, data_template)
	inputspec.inputs.file_list.sort()
	print(inputspec.inputs.file_list)
	print(len(inputspec.inputs.file_list))
	# inputspec.inputs.warp_list = files_from_template(template_subj_list, warp_template)
	inputspec.inputs.warp_list = files_from_template(subj_list, warp_template)
	# inputspec.inputs.warp_list = [i for i in inputspec.inputs.warp_list if '014' not in i]
	inputspec.inputs.warp_list.sort()
	print(inputspec.inputs.warp_list)
	print(len(inputspec.inputs.warp_list))
	# #to prevent missing files for a single subject.....
	# subj_list = [i.split("/")[-2] for i in inputspec.inputs.file_list]
	# print(subj_list)
	# rename files, as names are often indistinguishable (e.g. res4d.nii.gz)
	def rename_list(in_list):
		import nibabel as nib
		import os
		out_list = []
		for file in in_list:
			file_in = nib.load(file)
			nib.save(file_in, os.path.join(os.getcwd(), '_'.join(file.split('/')[-3:])))
			out_list.append(os.path.join(os.getcwd(), '_'.join(file.split('/')[-3:])))
		return out_list
	rename = pe.Node(Function(
		input_names=['in_list'],
		output_names=['out_list'],
			function=rename_list),
					name='rename')
	# dartel warping node.
	warp_data = pe.Node(interface=CreateWarped(), name='warp_data')
	#warp_data.inputs.image_files = # from inputspec OR gunzip
	#warp_data.inputs.flowfield_files = # from inputspec
	sinker = pe.Node(DataSink(), name='sinker')
	sinker.inputs.base_directory = out_dir
	# check if unzipping is necessary.
	apply_warp_wf.connect([(inputspec, rename, [('file_list', 'in_list')]),
						   (inputspec, warp_data, [('warp_list', 'flowfield_files')]),
						   (warp_data, sinker, [('warped_files', 'warped_files')])])
	if any('nii.gz' in file for file in files_from_template(subj_list, data_template)):
		from nipype.algorithms.misc import Gunzip
		gunzip = pe.MapNode(interface=Gunzip(), name='gunzip', iterfield=['in_file'])
		apply_warp_wf.connect([(rename, gunzip, [('out_list', 'in_file')]),
							   (gunzip, warp_data, [('out_file', 'image_files')])])
	else:
		apply_warp_wf.connect([(rename, warp_data, [('out_list', 'image_files')])])
	return apply_warp_wf

def files_from_template(identity_list, template):
	'''
	Uses glob to grab all matches to a template, then subsets the list with identifier.
	Input [Mandatory]:
		identity_list: string or list of strings, to grab subset of glob search.
			e.g. 'sub-01', or ['sub-01', 'sub-02', 'sub-03']
		template: string denoting a path, with wildcards, to be used in glob.
			e.g. '/home/neuro/data/smoothsub-*/model/sub-*.nii.gz'
	Output:
		out_list: list of file paths, first from the glob template,
		then subseted by identifier.
	'''
	import glob
	out_list = []
	if type(identity_list) != list:
		assert (type(identity_list) == str), 'identifier must be either a string, or a list of string'
		identity_list = [identity_list]
	for x in glob.glob(template):
		if any(subj in x for subj in identity_list):
			out_list.append(x)
	return out_list
 
import os, glob
from nilearn.image import resample_img
import nibabel as nib
import pandas as pd
import numpy as np
import os
import nibabel as nib
import glob
import math

subj_list = os.environ.get("SUBJ_LIST").split(" ")
print(subj_list)
print(len(subj_list))
subj_existed = [i.split('/')[-1].split('w_')[-1].split('__')[0] for i in glob.glob('/scratch/output/3_passive/warped_files/*')]
subj_list = [i for i in subj_list if i not in subj_existed]
print(subj_list)
print(len(subj_list))

for s in subj_list:
	subj = s[0:7]
	run = s[-6:]
	print('working on: '+subj+'_'+run)

	# subj_df_original = pd.read_table(glob.glob('/scratch/wrkdir/event_files/'+subj+'/func/'+subj+'*3_'+run+'_events.tsv')[0], sep='\t')
	# subj_df_event = subj_df_original[(subj_df_original['trial_type'] == 'CS') | (subj_df_original['trial_type'] == 'US') | (subj_df_original['trial_type'] == 'decision')]
	# subj_df_event = subj_df_event.reset_index(drop=True)
	# subj_df_event.loc[(subj_df_event['participant_made_choice']=='computer') & (subj_df_event['trial_type']=='decision'), 'trial_type'] = 'passive'
	# for i in range(len(subj_df_event['RT'])):
	# 	try:
	# 		subj_df_event['RT'][i] = float(subj_df_event['RT'][i])
	# 	except:
	# 		subj_df_event['RT'][i] = np.nan
	# subj_df_event.loc[(subj_df_event['participant_made_choice']=='participant') & (subj_df_event['trial_type']=='decision') & (pd.isnull(subj_df_event['RT'])), 'trial_type'] = 'active_no_motor'
	# subj_df_event.loc[(subj_df_event['participant_made_choice']=='participant') & (subj_df_event['trial_type']=='decision') & (pd.notnull(subj_df_event['RT'])), 'trial_type'] = 'active_motor'
	# if np.sum(subj_df_event['trial_type']=='active_no_motor')>0 and np.sum(subj_df_event['trial_type']=='active_motor')>0:
	# {'1_CS': '/scratch/data/'+s+'/cope1.nii.gz',
	# 2_US': '/scratch/data/'+s+'/cope2.nii.gz'}
	if len(glob.glob('/scratch/data/'+s+'/cope*.nii.gz'))==9 and not os.path.exists('/scratch/output/9_active_no_motor_passive_contrast/warped_files/wdata_'+s+'_cope9.nii.gz'):
		print('both')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


		data_templates = {'3_passive': '/scratch/data/'+s+'/cope3.nii.gz',
						  '4_active_no_motor': '/scratch/data/'+s+'/cope4.nii.gz',
						  '5_active_motor': '/scratch/data/'+s+'/cope5.nii.gz',
						  '6_active_motor_RT': '/scratch/data/'+s+'/cope6.nii.gz',
						  '7_active_motor_no_motor_contrast': '/scratch/data/'+s+'/cope7.nii.gz',
						  '8_active_motor_passive_contrast': '/scratch/data/'+s+'/cope8.nii.gz',
						  '9_active_no_motor_passive_contrast': '/scratch/data/'+s+'/cope9.nii.gz'}

	# elif np.sum(subj_df_event['trial_type']=='active_no_motor')==0 and np.sum(subj_df_event['trial_type']=='active_motor')>0:
	elif len(glob.glob('/scratch/data/'+s+'/cope*.nii.gz'))==6 and not os.path.exists('/scratch/output/8_active_motor_passive_contrast/warped_files/wdata_'+s+'_cope8.nii.gz'):
		print('just motor')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


		print('renaming')
		os.rename('/scratch/data/'+s+'/cope6.nii.gz', '/scratch/data/'+s+'/cope8.nii.gz')
		os.rename('/scratch/data/'+s+'/cope5.nii.gz', '/scratch/data/'+s+'/cope6.nii.gz')
		os.rename('/scratch/data/'+s+'/cope4.nii.gz', '/scratch/data/'+s+'/cope5.nii.gz')
        
		data_templates = {'3_passive': '/scratch/data/'+s+'/cope3.nii.gz',
				  '5_active_motor': '/scratch/data/'+s+'/cope5.nii.gz',
				  '6_active_motor_RT': '/scratch/data/'+s+'/cope6.nii.gz',
				  '8_active_motor_passive_contrast': '/scratch/data/'+s+'/cope8.nii.gz'}

	# elif np.sum(subj_df_event['trial_type']=='active_no_motor')>0 and np.sum(subj_df_event['trial_type']=='active_motor')==0:
	elif len(glob.glob('/scratch/data/'+s+'/cope*.nii.gz'))==5 and not os.path.exists('/scratch/output/9_active_no_motor_passive_contrast/warped_files/wdata_'+s+'_cope9.nii.gz'):
		print('just no motor')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

		print('renaming')
		os.rename('/scratch/data/'+s+'/cope5.nii.gz', '/scratch/data/'+s+'/cope9.nii.gz')

		data_templates = {'3_passive': '/scratch/data/'+s+'/cope3.nii.gz',
				  '4_active_no_motor': '/scratch/data/'+s+'/cope4.nii.gz',
				  '9_active_no_motor_passive_contrast': '/scratch/data/'+s+'/cope9.nii.gz'}

	else:
		continue

	work_dir = '/scratch/wrkdir'
	out_dir = '/scratch/output'
	# for f in glob.glob(os.path.join('/scratch/wrkdir/dartel_flow', 'u_SC*sub-*_Template.nii')):
	# 	print('resampling')
	# 	f_resample = resample_img(nib.load(f),
	# 			target_affine=nib.load(glob.glob('/scratch/data/*/cope1.nii.gz')[0]).affine,
	# 			target_shape=nib.load(glob.glob('/scratch/data/*/cope1.nii.gz')[0]).shape[0:3],
	# 			interpolation='nearest').get_fdata()
	# 	# f_resample = f_resample[:,:,:,0,0]
	# 	f_resample = nib.Nifti1Image(f_resample, nib.load(glob.glob('/scratch/data/*/cope1.nii.gz')[0]).affine, nib.load(glob.glob('/scratch/data/*/cope1.nii.gz')[0]).header)
	# 	print(f_resample.shape)
	# 	print(nib.load(glob.glob('/scratch/data/*/cope1.nii.gz')[0]).shape)
	# 	nib.save(f_resample, f.split('.nii')[0]+'_resampled.nii')
	warp_template = os.path.join('/scratch/dartel_flow', 'u_SC*'+subj+'_'+run+'*_Template_resampled.nii')

	for data in data_templates:
		print("********************************************************")
		print(data)
		print("********************************************************")
		DARTEL_warp = setup_DARTEL_warp_wf(subj, data_templates[data],
										   warp_template, os.path.join(work_dir, data), os.path.join(out_dir, data))
		DARTEL_warp.run(plugin='MultiProc')

# import os, glob
# from nilearn.image import resample_img
# import nibabel as nib

# for f in glob.glob('/scratch/data/painAvd/*/*.nii.gz'):
# 	print(f)
# 	f_resample = resample_img(nib.load(f),
# 			target_affine=nib.load(glob.glob('/scratch/template/cope1.nii.gz')[0]).affine,
# 			target_shape=nib.load(glob.glob('/scratch/template/cope1.nii.gz')[0]).shape[0:3],
# 			interpolation='nearest').get_fdata()
# 	# f_resample = f_resample[:,:,:,0,0]
# 	f_resample = nib.Nifti1Image(f_resample, nib.load(glob.glob('/scratch/template/cope1.nii.gz')[0]).affine, nib.load(glob.glob('/scratch/template/cope1.nii.gz')[0]).header)
# 	print(f_resample.shape)
# 	print(nib.load(glob.glob('/scratch/template/cope1.nii.gz')[0]).shape)
# 	nib.save(f_resample, f.split('.nii')[0]+'_resampled.nii.gz')





