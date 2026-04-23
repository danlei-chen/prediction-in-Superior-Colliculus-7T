import os, sys
import nipype.pipeline.engine as pe
from nipype import IdentityInterface

def create_lvl1pipe_wf(options):
    import nipype.pipeline.engine as pe
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface, SelectFiles
    from nipype.interfaces.utility.wrappers import Function

    lvl1pipe_wf = pe.Workflow(name='lvl_one_pipe')

    inputspec = pe.Node(IdentityInterface(
        fields=['input_dir', 'output_dir', 'design_col', 'noise_regressors',
                'noise_transforms', 'TR', 'FILM_threshold', 'hpf_cutoff',
                'conditions', 'contrasts', 'bases', 'model_serial_correlations',
                'sinker_subs', 'bold_template', 'mask_template', 'task_template',
                'confound_template', 'smooth_gm_mask_template', 'gmmask_args',
                'subject_id', 'fwhm', 'proj_name'],
        mandatory_inputs=False),
        name='inputspec')

    def get_file(subj_id, template):
        import glob
        temp_list = []
        out_list = []
        if '_' in subj_id and '/anat/' in list(template.values())[0]:
            subj_id = subj_id[:subj_id.find('_')]
        for x in glob.glob(list(template.values())[0]):
            if subj_id in x:
                temp_list.append(x)
        for file in temp_list:
            if file not in out_list:
                out_list.append(file)
        if len(out_list) == 0:
            raise ValueError(f'No files found for {subj_id}')
        if len(out_list) > 1:
            raise ValueError(f'Multiple files found for {subj_id}: {len(out_list)}')
        return out_list[0]

    get_bold = pe.Node(Function(input_names=['subj_id', 'template'],
        output_names=['out_file'], function=get_file), name='get_bold')
    get_mask = pe.Node(Function(input_names=['subj_id', 'template'],
        output_names=['out_file'], function=get_file), name='get_mask')
    get_task = pe.Node(Function(input_names=['subj_id', 'template'],
        output_names=['out_file'], function=get_file), name='get_task')
    get_confile = pe.Node(Function(input_names=['subj_id', 'template'],
        output_names=['out_file'], function=get_file), name='get_confile')

    if options['smooth']:
        get_gmmask = pe.Node(Function(input_names=['subj_id', 'template'],
            output_names=['out_file'], function=get_file), name='get_gmmask')
        mod_gmmask = pe.Node(fsl.maths.MathsCommand(), name='mod_gmmask')

        def fit_mask(mask_file, ref_file):
            from nilearn.image import resample_img
            import nibabel as nib
            import os
            out_file = resample_img(nib.load(mask_file),
                                   target_affine=nib.load(ref_file).affine,
                                   target_shape=nib.load(ref_file).shape[0:3],
                                   interpolation='nearest')
            nib.save(out_file, os.path.join(os.getcwd(), mask_file.split('.nii')[0]+'_fit.nii.gz'))
            out_mask = os.path.join(os.getcwd(), mask_file.split('.nii')[0]+'_fit.nii.gz')
            return out_mask

        fit_mask = pe.Node(Function(input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'], function=fit_mask), name='fit_mask')

    def get_terms(confound_file, noise_transforms, noise_regressors, TR, options):
        import numpy as np
        import pandas as pd
        from nltools.data import Design_Matrix

        df_cf = pd.DataFrame(pd.read_csv(confound_file, sep='\t'))
        transfrm_list = []
        for idx, entry in enumerate(noise_regressors):
            if '*' in entry:
                transfrm_list.append(entry.replace('*', ''))
                noise_regressors[idx] = entry.replace('*', '')
        confounds = df_cf[noise_regressors]
        transfrmd_cnfds = df_cf[transfrm_list]
        TR_time = pd.Series(np.arange(0.0, TR*transfrmd_cnfds.shape[0], TR))
        
        if options.get('spike_regression'):
            fd = df_cf['FramewiseDisplacement']
            for spike in fd.index[fd > options['spike_regression']].tolist():
                spike_vec = np.zeros(fd.shape[0])
                spike_vec[spike] = 1
                confounds = confounds.join(pd.DataFrame(columns=['spike_'+str(spike)], 
                    data={'spike_'+str(spike):spike_vec}))
        
        if 'quad' in noise_transforms:
            quad = np.square(transfrmd_cnfds - np.mean(transfrmd_cnfds, axis=0))
            confounds = confounds.join(quad, rsuffix='_quad')
        if 'tderiv' in noise_transforms:
            tderiv = pd.DataFrame(pd.Series(np.gradient(transfrmd_cnfds[col]), TR_time)
                                  for col in transfrmd_cnfds).T
            tderiv.columns = transfrmd_cnfds.columns
            tderiv.index = confounds.index
            confounds = confounds.join(tderiv, rsuffix='_tderiv')
        if 'quadtderiv' in noise_transforms:
            quadtderiv = np.square(tderiv)
            confounds = confounds.join(quadtderiv, rsuffix='_quadtderiv')
        
        if options.get('remove_steadystateoutlier'):
            if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]])
            elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]])
        
        if options.get('ICA_AROMA'):
            if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]])
            elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]])
        
        confounds = Design_Matrix(confounds, sampling_freq=1/TR)
        if isinstance(options.get('poly_trend'), int):
            confounds = confounds.add_poly(order=options['poly_trend'])
        if isinstance(options.get('dct_basis'), int):
            confounds = confounds.add_dct_basis(duration=options['dct_basis'])
        
        return confounds

    get_confounds = pe.Node(Function(input_names=['confound_file', 'noise_transforms',
                                                  'noise_regressors', 'TR', 'options'],
                                 output_names=['confounds'], function=get_terms),
                         name='get_confounds')
    get_confounds.inputs.options = options

    def get_subj_info(task_file, design_col, confounds, conditions):
        from nipype.interfaces.base import Bunch
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import scale

        onsets = []
        durations = []
        amplitudes = []
        df = pd.read_csv(task_file, sep='\t', parse_dates=False)
        
        for idx, cond in enumerate(conditions):
            if isinstance(cond, list):
                c = cond[2] == 'cent'
                n = cond[3] == 'norm'
                onsets.append(list(df[df[design_col] == cond[0]].onset))
                durations.append(list(df[df[design_col] == cond[0]].duration))
                amp_temp = list(scale(df[df[design_col] == cond[0]][cond[1]].tolist(),
                                   with_mean=c, with_std=n))
                amp_temp = pd.Series(amp_temp, dtype=object).fillna(0).tolist()
                amplitudes.append(amp_temp)
                conditions[idx] = cond[0]+'_'+cond[1]
            elif isinstance(cond, str):
                onsets.append(list(df[df[design_col] == cond].onset))
                durations.append(list(df[df[design_col] == cond].duration))
                amplitudes.append(list(np.repeat(1, len(df[df[design_col] == cond].onset))))

        output = Bunch(conditions=conditions, onsets=onsets, durations=durations,
                       amplitudes=amplitudes, tmod=None, pmod=None,
                       regressor_names=confounds.columns.values,
                       regressors=confounds.T.values.tolist())
        return output

    make_bunch = pe.Node(Function(input_names=['task_file', 'design_col', 'confounds', 'conditions'],
                                 output_names=['subject_info'], function=get_subj_info),
                         name='make_bunch')

    def mk_outdir(output_dir, options, proj_name):
        import os
        prefix = proj_name
        if options['smooth']:
            new_out_dir = os.path.join(output_dir, prefix, 'smooth')
        else:
            new_out_dir = os.path.join(output_dir, prefix, 'nosmooth')
        if not os.path.isdir(new_out_dir):
            os.makedirs(new_out_dir)
        return new_out_dir

    make_outdir = pe.Node(Function(input_names=['output_dir', 'options', 'proj_name'],
                                   output_names=['new_out_dir'], function=mk_outdir),
                          name='make_outdir')
    make_outdir.inputs.options = options

    def mask_img(img_file, mask_file):
        import numpy as np
        import nibabel as nib
        import os.path
        from skimage.transform import resize
        mask = nib.load(mask_file)
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
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(out_img, os.path.join(os.getcwd(), 'masked_bold.nii.gz'))
        return os.path.join(os.getcwd(), 'masked_bold.nii.gz')

    maskBold = pe.Node(Function(input_names=['img_file', 'mask_file'],
                                output_names=['out_file'], function=mask_img),
                      name='maskBold')

    from nipype.interfaces.afni import Despike
    despike = pe.Node(Despike(), name='despike')
    despike.inputs.outputtype = 'NIFTI_GZ'

    from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
    smooth_wf = create_susan_smooth()

    import nipype.algorithms.modelgen as model
    specify_model = pe.Node(interface=model.SpecifyModel(), name='specify_model')
    specify_model.inputs.input_units = 'secs'

    from nipype.workflows.fmri.fsl import estimate
    modelfit = estimate.create_modelfit_workflow()
    modelfit.base_dir = '.'

    if not options['run_contrasts']:
        modelestimate = modelfit.get_node('modelestimate')
        merge_contrasts = modelfit.get_node('merge_contrasts')
        ztop = modelfit.get_node('ztop')
        outputspec = modelfit.get_node('outputspec')
        modelfit.disconnect([(modelestimate, merge_contrasts, [('zstats', 'in1'), ('zfstats', 'in2')]),
                             (merge_contrasts, ztop, [('out', 'in_file')]),
                             (merge_contrasts, outputspec, [('out', 'zfiles')]),
                             (ztop, outputspec, [('out_file', 'pfiles')])])
        modelfit.remove_nodes([merge_contrasts, ztop])

    from nipype.interfaces.io import DataSink
    sinker = pe.Node(DataSink(), name='sinker')

    def negate(input):
        return not input

    def unlist(input):
        return input[0]

    lvl1pipe_wf.connect([
        (inputspec, get_bold, [('subject_id', 'subj_id'), ('bold_template', 'template')]),
        (inputspec, get_mask, [('subject_id', 'subj_id'), ('mask_template', 'template')]),
        (inputspec, get_task, [('subject_id', 'subj_id'), ('task_template', 'template')]),
        (inputspec, get_confile, [('subject_id', 'subj_id'), ('confound_template', 'template')]),
        (inputspec, get_confounds, [('noise_transforms', 'noise_transforms'),
                                     ('noise_regressors', 'noise_regressors'), ('TR', 'TR')]),
        (inputspec, make_bunch, [('design_col', 'design_col'), ('conditions', 'conditions')]),
        (inputspec, make_outdir, [('output_dir', 'output_dir'), ('proj_name', 'proj_name')]),
        (inputspec, specify_model, [('hpf_cutoff', 'high_pass_filter_cutoff'), ('TR', 'time_repetition')]),
        (inputspec, modelfit, [('TR', 'inputspec.interscan_interval'),
                                ('FILM_threshold', 'inputspec.film_threshold'),
                                ('bases', 'inputspec.bases'),
                                ('model_serial_correlations', 'inputspec.model_serial_correlations'),
                                (('model_serial_correlations', negate), 'modelestimate.autocorr_noestimate'),
                                ('contrasts', 'inputspec.contrasts')]),
        (get_confile, get_confounds, [('out_file', 'confound_file')]),
        (get_confounds, make_bunch, [('confounds', 'confounds')]),
        (get_task, make_bunch, [('out_file', 'task_file')]),
        (make_bunch, specify_model, [('subject_info', 'subject_info')]),
        (get_mask, maskBold, [('out_file', 'mask_file')]),
    ])

    if options['censoring'] == 'despike':
        lvl1pipe_wf.connect([(get_bold, despike, [('out_file', 'in_file')])])
        if options['smooth']:
            lvl1pipe_wf.connect([
                (inputspec, smooth_wf, [('fwhm', 'inputnode.fwhm')]),
                (inputspec, get_gmmask, [('subject_id', 'subj_id'), ('smooth_gm_mask_template', 'template')]),
                (get_gmmask, mod_gmmask, [('out_file', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (mod_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (fit_mask, smooth_wf, [('out_mask', 'inputnode.mask_file')]),
                (fit_mask, sinker, [('out_mask', 'smoothing_mask')]),
                (despike, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [(('outputnode.smoothed_files', unlist), 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
            ])
        else:
            lvl1pipe_wf.connect([
                (get_bold, maskBold, [('out_file', 'img_file')]),
                (despike, specify_model, [('out_file', 'functional_runs')]),
                (despike, modelfit, [('out_file', 'inputspec.functional_data')]),
                (despike, sinker, [('out_file', 'despike')])
            ])
    else:
        if options['smooth']:
            lvl1pipe_wf.connect([
                (inputspec, smooth_wf, [('fwhm', 'inputnode.fwhm')]),
                (inputspec, get_gmmask, [('subject_id', 'subj_id'), ('smooth_gm_mask_template', 'template')]),
                (get_gmmask, mod_gmmask, [('out_file', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (mod_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (fit_mask, smooth_wf, [('out_mask', 'inputnode.mask_file')]),
                (fit_mask, sinker, [('out_mask', 'smoothing_mask')]),
                (get_bold, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [(('outputnode.smoothed_files', unlist), 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
            ])
        else:
            lvl1pipe_wf.connect([
                (get_bold, maskBold, [('out_file', 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
            ])

    lvl1pipe_wf.connect([
        (specify_model, modelfit, [('session_info', 'inputspec.session_info')]),
        (inputspec, sinker, [('subject_id', 'container'), ('sinker_subs', 'substitutions')]),
        (make_outdir, sinker, [('new_out_dir', 'base_directory')]),
        (modelfit, sinker, [('outputspec.parameter_estimates', 'model'),
                            ('outputspec.dof_file', 'model.@dof'),
                            ('outputspec.copes', 'model.@copes'),
                            ('outputspec.varcopes', 'model.@varcopes'),
                            ('outputspec.zfiles', 'stats'),
                            ('outputspec.pfiles', 'stats.@pfiles'),
                            ('level1design.ev_files', 'design'),
                            ('level1design.fsf_files', 'design.@fsf'),
                            ('modelgen.con_file', 'design.@confile'),
                            ('modelgen.fcon_file', 'design.@fconfile'),
                            ('modelgen.design_cov', 'design.@covmatriximg'),
                            ('modelgen.design_image', 'design.@designimg'),
                            ('modelgen.design_file', 'design.@designfile'),
                            ('modelestimate.logfile', 'design.@log'),
                            ('modelestimate.sigmasquareds', 'model.@resid_sum'),
                            ('modelestimate.fstats', 'stats.@fstats'),
                            ('modelestimate.thresholdac', 'model.@serial_corr')])
    ])
    
    if options['keep_resid']:
        lvl1pipe_wf.connect([(modelfit, sinker, [('modelestimate.residual4d', 'model.@resid')])])
    
    return lvl1pipe_wf

# ========== EXECUTION ==========

sinker_subs = [('subject_id_', ''), ('_fwhm', 'fwhm'), ('subject_id_sub', 'sub'), (' ', '_')]

options = {
    'remove_steadystateoutlier': True,
    'smooth': False,
    'censoring': '',
    'ICA_AROMA': False,
    'poly_trend': 0,
    'dct_basis': 120,
    'run_contrasts': True,
    'keep_resid': True,
    'spike_regression': 0.5}

model_wf = create_lvl1pipe_wf(options)
model_wf.inputs.inputspec.input_dir = '/data/bids/'
model_wf.inputs.inputspec.output_dir = '/output/analysis/'
model_wf.inputs.inputspec.design_col = 'trial_type'
model_wf.inputs.inputspec.noise_regressors = ['X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*',
                                               'aCompCorCSF0', 'aCompCorCSF1', 'aCompCorCSF2',
                                               'aCompCorCSF3', 'aCompCorCSF4', 'aCompCorWM0',
                                               'aCompCorWM1', 'aCompCorWM2', 'aCompCorWM3', 'aCompCorWM4']
model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
model_wf.inputs.inputspec.TR = 2.34
model_wf.inputs.inputspec.FILM_threshold = 1
model_wf.inputs.inputspec.hpf_cutoff = 0.0

model_wf.inputs.inputspec.conditions = ['shape', 'decision', 'stimulation']
model_wf.inputs.inputspec.contrasts = [['shape', 'T', ['shape'], [1]],
                                       ['decision', 'T', ['decision'], [1]],
                                       ['stimulation', 'T', ['stimulation'], [1]]]
model_wf.inputs.inputspec.bases = {'dgamma': {'derivs': False}}
model_wf.inputs.inputspec.model_serial_correlations = True
model_wf.inputs.inputspec.sinker_subs = sinker_subs
model_wf.inputs.inputspec.bold_template = {'bold': '/data/bids/sub-*/func/*_bold_space-MNI_preproc.nii.gz'}
model_wf.inputs.inputspec.mask_template = {'mask': '/data/bids/sub-*/func/*_bold_brainmask.nii.gz'}
model_wf.inputs.inputspec.task_template = {'task': '/data/bids/sub-*/func/*_events.tsv'}
model_wf.inputs.inputspec.confound_template = {'confound': '/data/bids/sub-*/func/*_confounds.tsv'}
model_wf.inputs.inputspec.smooth_gm_mask_template = {'gm_mask': '/data/bids/sub-*/anat/*_class-GM_probtissue.nii.gz'}
model_wf.inputs.inputspec.gmmask_args = '-thr .5 -bin -kernel gauss 1 -dilM'
model_wf.inputs.inputspec.proj_name = 'project_analysis'

subject_list = ['sub-001', 'sub-002']
fwhm_list = []

infosource = pe.Node(IdentityInterface(fields=['fwhm', 'subject_id']), name='infosource')
infosource.iterables = [('subject_id', subject_list)]

full_model_wf = pe.Workflow(name='full_model_wf')
full_model_wf.connect([(infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])

full_model_wf.base_dir = '/work/analysis'
full_model_wf.crash_dump = '/work/crash'

full_model_wf.run(plugin='MultiProc', plugin_args={'n_procs': 2})