import os, sys
import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface
def create_lvl1pipe_wf(options):
    '''
    Input [Mandatory]:
        ~~~~~~~~~~~ Set in command call:

        options: dictionary with the following entries
            remove_steadystateoutlier [boolean]:
                Should always be True. Remove steady state outliers from bold timecourse, specified in fmriprep confounds file.
            smooth [boolean]:
                If True, then /smooth subfolder created and populated with results. If False, then /nosmooth subfolder created and populated with results.
            censoring [string]:
                Either '' or 'despike', which implements nipype.interfaces.afni.Despike
            ICA_AROMA [boolean]:
                Use AROMA error components, from fmriprep confounds file.
            run_contrasts [boolean]:
                If False, then components related to contrasts and p values are removed from   nipype.workflows.fmri.fsl.estimate.create_modelfit_workflow()
            keep_resid [boolean]:
                If False, then only sum of squares residuals will be outputted. If True, then timecourse residuals kept.
            poly_trend [integer. Use None to skip]:
                If given, polynomial trends will be added to run confounds, up to the order of the integer
                e.g. "0", gives an intercept, "1" gives intercept + linear trend,
                "2" gives intercept + linear trend + quadratic.
                DO NOT use in conjnuction with high pass filters.
            dct_basis [integer. Use None to skip]:
                If given, adds a discrete cosine transform, with a length (in seconds) of the interger specified.
                    Adds unit scaled cosine basis functions to Design_Matrix columns,
                    based on spm-style discrete cosine transform for use in
                    high-pass filtering. Does not add intercept/constant.
                    DO NOT use in conjnuction with high pass filters.
            spike_regression [float]
                If given, all TRs with framewise displacement above this value are
                modeled as single impulse spikes

        ~~~~~~~~~~~ Set through inputs.inputspec

        input_dir [string]:
            path to folder containing fmriprep preprocessed data.
            e.g. model_wf.inputs.inputspec.input_dir = '/home/neuro/data'
        output_dir [string]:
            path to desired output folder. Workflow will create a new subfolder based on proj_name.
            e.g. model_wf.inputs.inputspec.output_dir = '/home/neuro/output'
        proj_name [string]:
            name for project subfolder within output_dir. Ideally something unique, or else workflow will write to an existing folder.
            e.g. model_wf.inputs.inputspec.proj_name = 'FSMAP_stress'
        design_col [string]:
            Name of column within events.tsv with values corresponding to entries specified in params.
            e.g. model_wf.inputs.inputspec.design_col = 'trial_type'
        params [list fo strings]:
            values within events.tsv design_col that correspond to events to be modeled.
            e.g.  ['Instructions', 'Speech_prep', 'No_speech']
        conditions [list, of either strings or lists],
            each condition must be a string within the events.tsv design_col.
            These conditions correspond to event conditions to be modeled.
            Give a list, instead of a string, to model parametric terms.
            These parametric terms give a event condition, then a parametric term, which is another column in the events.tsv file.
            The parametric term can be centered and normed using entries 3 and 4 in the list.
            e.g. model_wf.inputs.inputspec.params = ['condition1',
                                                     'condition2',
                                                    ['condition1', 'parametric1', 'no_cent', 'no_norm'],
                                                    ['condition2', 'paramatric2', 'cent', 'norm']]
                     entry 1 is a condition within the design_col column
                     entry 2 is a column in the events folder, which will be used for parametric weightings.
                     entry 3 is either 'no_cent', or 'cent', indicating whether to center the parametric variable.
                     entry 4 is either 'no_norm', or 'norm', indicating whether to normalize the parametric variable.
             Onsets and durations will be taken from corresponding values for entry 1
             parametric weighting specified by entry 2, scaled/centered as specified, then
             appended to the design matrix.
        contrasts [list of lists]:
            Specifies contrasts to be performed. using params selected above.
            e.g. model_wf.inputs.inputspec.contrasts =
                [['Instructions', 'T', ['Instructions'], [1]],
                 ['Speech_prep', 'T', ['Speech_prep'], [1]],
                 ['No_speech', 'T', ['No_speech'], [1]],
                 ['Speech_prep>No_speech', 'T', ['Speech_prep', 'No_speech'], [1, -1]]]
        noise_regressors [list of strings]:
            column names in confounds.tsv, specifying desired noise regressors for model.
            IF noise_transforms are to be applied to a regressor, add '*' to the name.
            e.g. model_wf.inputs.inputspec.noise_regressors = ['CSF', 'WhiteMatter', 'GlobalSignal', 'X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*']
        noise_transforms [list of strings]:
            noise transforms to be applied to select noise_regressors above. Possible values are 'quad', 'tderiv', and 'quadtderiv', standing for quadratic function of value, temporal derivative of value, and quadratic function of temporal derivative.
            e.g. model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
        TR [float]:
            Scanner TR value in seconds.
            e.g. model_wf.inputs.inputspec.TR = 2.
        FILM_threshold [integer]:
            Cutoff value for modeling threshold. 1000: p <.001; 1: p <=1, i.e. unthresholded.
            e.g. model_wf.inputs.inputspec.FILM_threshold = 1
        hpf_cutoff [float]:
            high pass filter value. DO NOT USE THIS in conjunction with poly_trend or dct_basis.
            e.g. model_wf.inputs.inputspec.hpf_cutoff = 120.
        bases: (a dictionary with keys which are 'hrf' or 'fourier' or 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
             dict {'name':{'basesparam1':val,...}}
             name : string
             Name of basis function (hrf, fourier, fourier_han, gamma, fir)
             hrf :
                 derivs : 2-element list
                    Model HRF Derivatives. No derivatives: [0,0],
                    Time derivatives : [1,0],
                    Time and Dispersion derivatives: [1,1]
             fourier, fourier_han, gamma, fir:
                 length : int
                    Post-stimulus window length (in seconds)
                 order : int
                    Number of basis functions
            e.g. model_wf.inputs.inputspec.bases = {'dgamma':{'derivs': False}}
        model_serial_correlations [boolean]:
            Allow prewhitening, with 5mm spatial smoothing.
            model_wf.inputs.inputspec.model_serial_correlations = True
        sinker_subs [list of tuples]:
            passed to nipype.interfaces.io.Datasink. Changes names when passing to output directory.
            e.g. model_wf.inputs.inputspec.sinker_subs =
                [('pe1', 'pe1_instructions'),
                 ('pe2', 'pe2_speech_prep'),
                 ('pe3', 'pe3_no_speech')]
        bold_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant BOLD files. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.bold_template =
                {'bold': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'}
                 This would grab the functional run for all subjects, and when subject_id = 'sub-001', there is ONE file in the list that the ID could possible correspond to.
                To handle multiple runs, list the run information in the subject_id. e.g. 'sub-01_task-trag_run-01'.
        mask_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant MASK files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.mask_template =
            {'mask': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'}
            See bold_template for more detail.
        task_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant events.tsv files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.task_template =
            {'task': '/home/neuro/data/sub-*/func/sub-*_task-stress_events.tsv'}
            See bold_template for more detail.
        confound_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant confounds.tsv files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.confound_template =
            {'confound': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_confounds.tsv'}
            See bold_template for more detail.
        smooth_gm_mask_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant grey matter mask .nii.gz files, pulling from each subject's /anat fodler. Each subject_list entry should uniquely identify the ONE relevant file (BUT SEE THE NOTE BELOW).
            e.g. model_wf.inputs.inputspec.smooth_gm_mask_template =
                {'gm_mask': '/scratch/data/sub-*/anat/sub-*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz'}
                NOTE: If the subject_id value has more information than just the ID (e.g. sub-01_task-trag_run-01), then JUST the sub-01 portion will be used to identify the grey matter mask. This is because multiple runs will have the same anatomical data. i.e. sub-01_run-01, sub-01_run-02, sub-01_run-03, all correspond to sub-01_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz.
        fwhm [float]. Redundant if options['smooth']: False
            Determines smoothing kernel. Multiple kernels can be run in parallel by iterating through an outside workflow. Also see subject_id below for another example of iterables.
            e.g.
                model_wf.inputs.inputspec.fwhm = 1.5
            OR Iterable e.g.
                import nipype.pipeline.engine as pe
                fwhm_list = [1.5, 6]
                infosource = pe.Node(IdentityInterface(fields=['fwhm']),
                           name='infosource')
                infosource.iterables = [('fwhm', fwhm_list)]
                full_model_wf = pe.Workflow(name='full_model_wf')
                full_model_wf.connect([(infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])
                full_model_wf.run()
        subject_id [string]:
            Identifies subject in conjnuction with template. See bold_template note above.
            Can also be entered as an iterable from an outside workflow, in which case iterables are run in parallel to the extent that cpu cores are available.
            e.g.
                model_wf.inputs.inputspec.subject_id = 'sub-01'
            OR Iterable e.g.
                import nipype.pipeline.engine as pe
                subject_list = ['sub-001', 'sub-002']
                infosource = pe.Node(IdentityInterface(fields=['subject_id']),
                           name='infosource')
                infosource.iterables = [('subject_id', subject_list)]
                full_model_wf = pe.Workflow(name='full_model_wf')
                full_model_wf.connect([(infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])
                full_model_wf.run()
    '''
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface, SelectFiles
    from nipype.interfaces.utility.wrappers import Function

    ##################  Setup workflow.
    lvl1pipe_wf = pe.Workflow(name='lvl_one_pipe')

    inputspec = pe.Node(IdentityInterface(
        fields=['input_dir',
                'output_dir',
                'design_col',
                'noise_regressors',
                'noise_transforms',
                'TR', # in seconds.
                'FILM_threshold',
                'hpf_cutoff',
                'conditions',
                'contrasts',
                'bases',
                'model_serial_correlations',
                'sinker_subs',
                'bold_template',
                'mask_template',
                'task_template',
                'confound_template',
                'smooth_gm_mask_template',
                'gmmask_args',
                'subject_id',
                'fwhm',
                'proj_name',
                ],
        mandatory_inputs=False),
                 name='inputspec')

    ################## Select Files
    def get_file(subj_id, template):
        import glob
        temp_list = []
        out_list = []
        if '_' in subj_id and '/anat/' in list(template.values())[0]:
            subj_id = subj_id[:subj_id.find('_')]
            # if looking for gmmask, and subj_id includes additional info (e.g. sub-001_task-trag_run-01) then just take the subject id component, as the run info will not be present for the anatomical data.
        for x in glob.glob(list(template.values())[0]):
            if subj_id in x:
                temp_list.append(x)
        for file in temp_list: # ensure no duplicate entries.
            if file not in out_list:
                out_list.append(file)
        if len(out_list) == 0:
            assert (len(out_list) == 1), 'Each combination of template and subject ID should return 1 file. 0 files were returned.'
        if len(out_list) > 1:
            assert (len(out_list) == 1), 'Each combination of template and subject ID should return 1 file. Multiple files returned.'
        out_file = out_list[0]
        return out_file

    get_bold = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_bold')
    get_mask = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_mask')
    get_task = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_task')
    get_confile = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_confile')
    # get_bold.inputs.subj_id # From inputspec
    # get_bold.inputs.templates # From inputspec
    if options['smooth']:
        get_gmmask = pe.Node(Function(
            input_names=['subj_id', 'template'],
            output_names=['out_file'],
            function=get_file),
                            name='get_gmmask')

        mod_gmmask = pe.Node(fsl.maths.MathsCommand(),
                                name='mod_gmmask')
        # mod_gmmask.inputs.in_file = # from get_gmmask
        # mod_gmmask.inputs.args = from inputspec
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

        fit_mask = pe.Node(Function(
            input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'],
            function=fit_mask),
                            name='fit_mask')

    ################## Setup confounds
    def get_terms(confound_file, noise_transforms, noise_regressors, TR, options):
        '''
        Gathers confounds (and transformations) into a pandas dataframe.

        Input [Mandatory]:
            confound_file [string]: path to confound.tsv file, given by fmriprep.
            noise_transforms [list of strings]:
                noise transforms to be applied to select noise_regressors above. Possible values are 'quad', 'tderiv', and 'quadtderiv', standing for quadratic function of value, temporal derivative of value, and quadratic function of temporal derivative.
                e.g. model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
            noise_regressors [list of strings]:
                column names in confounds.tsv, specifying desired noise regressors for model.
                IF noise_transforms are to be applied to a regressor, add '*' to the name.
                e.g. model_wf.inputs.inputspec.noise_regressors = ['CSF', 'WhiteMatter', 'GlobalSignal', 'X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*']
            TR [float]:
                Scanner TR value in seconds.
            options: dictionary with the following entries
                remove_steadystateoutlier [boolean]:
                    Should always be True. Remove steady state outliers from bold timecourse, specified in fmriprep confounds file.
                ICA_AROMA [boolean]:
                    Use AROMA error components, from fmriprep confounds file.
                poly_trend [integer. Use None to skip]:
                    If given, polynomial trends will be added to run confounds, up to the order of the integer
                    e.g. "0", gives an intercept, "1" gives intercept + linear trend,
                    "2" gives intercept + linear trend + quadratic.
                dct_basis [integer. Use None to skip]:
                    If given, adds a discrete cosine transform, with a length (in seconds) of the interger specified.
                        Adds unit scaled cosine basis functions to Design_Matrix columns,
                        based on spm-style discrete cosine transform for use in
                        high-pass filtering. Does not add intercept/constant.
                spike_regression [float]
                    If given, all TRs with framewise displacement above this value are
                    modeled as single impulse spikes
        '''
        import numpy as np
        import pandas as pd
        from nltools.data import Design_Matrix

        df_cf = pd.DataFrame(pd.read_csv(confound_file, sep='\t'))
        transfrm_list = []
        for idx, entry in enumerate(noise_regressors): # get entries marked with *, indicating they should be transformed.
            if '*' in entry:
                transfrm_list.append(entry.replace('*', '')) # add entry to transformation list if it has *.
                noise_regressors[idx] = entry.replace('*', '')
        confounds = df_cf[noise_regressors]
        transfrmd_cnfds = df_cf[transfrm_list] # for transforms
        TR_time = pd.Series(np.arange(0.0, TR*transfrmd_cnfds.shape[0], TR)) # time series for derivatives.
        if options['spike_regression']:
            fd = df_cf['FramewiseDisplacement']
            for spike in fd.index[fd > options['spike_regression']].tolist():
                spike_vec = np.zeros(fd.shape[0])
                spike_vec[spike] = 1
                confounds = confounds.join(pd.DataFrame(columns=['spike_'+str(spike)], data={'spike_'+str(spike):spike_vec}))
        if 'quad' in noise_transforms:
            quad = np.square(transfrmd_cnfds - np.mean(transfrmd_cnfds,axis=0))
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
        if options['remove_steadystateoutlier']:
            if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]])
            elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]]) # old syntax
        if options['ICA_AROMA']:
            if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]])
            elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]].empty:
                confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]]) # old syntax
        confounds = Design_Matrix(confounds, sampling_freq=1/TR)
        if isinstance(options['poly_trend'], int):
            confounds = confounds.add_poly(order = options['poly_trend']) # these do not play nice with high pass filters.
        if isinstance(options['dct_basis'], int):
            confounds = confounds.add_dct_basis(duration=options['dct_basis']) # these do not play nice with high pass filters.
        return confounds

    get_confounds = pe.Node(Function(input_names=['confound_file', 'noise_transforms',
                                                  'noise_regressors', 'TR', 'options'],
                                 output_names=['confounds'],
                                  function=get_terms),
                         name='get_confounds')
    # get_confounds.inputs.confound_file =  # From get_confile
    # get_confounds.inputs.noise_transforms =  # From inputspec
    # get_confounds.inputs.noise_regressors =  # From inputspec
    # get_confounds.inputs.TR =  # From inputspec
    get_confounds.inputs.options = options

    ################## Create bunch to run FSL first level model.
    def get_subj_info(task_file, design_col, confounds, conditions):
        '''
        Makes a Bunch, giving all necessary data about conditions, onsets, and durations to
            FSL first level model. Needs a task file to run.

        Inputs:
            task file [string], path to the subject events.tsv file, as per BIDS format.
            design_col [string], column name within task file, identifying event conditions to model.
            confounds [pandas dataframe], pd.df of confounds, gathered from get_confounds node.
            conditions [list],
                e.g. ['condition1',
                      'condition2',
                     ['condition1', 'parametric1', 'no_cent', 'no_norm'],
                     ['condition2', 'paramatric2', 'cent', 'norm']]
                     each string entry (e.g. 'condition1') specifies a event condition in the design_col column.
                     each list entry includes 4 strings:
                         entry 1 is a condition within the design_col column
                         entry 2 is a column in the events folder, which will be used for parametric weightings.
                         entry 3 is either 'no_cent', or 'cent', indicating whether to center the parametric variable.
                         entry 4 is either 'no_norm', or 'norm', indicating whether to normalize the parametric variable.
                 Onsets and durations will be taken from corresponding values for entry 1
                 parametric weighting specified by entry 2, scaled/centered as specified, then
                appended to the design matrix.
        '''
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
                if cond[2] == 'no_cent': # determine whether to center/scale
                    c = False
                elif cond[2] == 'cent':
                    c = True
                if cond[3] == 'no_norm':
                    n = False
                elif cond[3] == 'norm':
                    n = True
                # grab parametric terms.
                onsets.append(list(df[df[design_col] == cond[0]].onset))
                durations.append(list(df[df[design_col] == cond[0]].duration))
                amp_temp = list(scale(df[df[design_col] == cond[0]][cond[1]].tolist(),
                                   with_mean=c, with_std=n)) # scale
                amp_temp = pd.Series(amp_temp, dtype=object).fillna(0).tolist() # fill na
                amplitudes.append(amp_temp) # append
                conditions[idx] = cond[0]+'_'+cond[1] # combine condition/parametric names and replace.
            elif isinstance(cond, str):
                onsets.append(list(df[df[design_col] == cond].onset))
                durations.append(list(df[df[design_col] == cond].duration))
                # dummy code 1's for non-parametric conditions.
                amplitudes.append(list(np.repeat(1, len(df[df[design_col] == cond].onset))))
            else:
                print('cannot identify condition:', cond)
        #             return None
        output = Bunch(conditions= conditions,
                           onsets=onsets,
                           durations=durations,
                           amplitudes=amplitudes,
                           tmod=None,
                           pmod=None,
                           regressor_names=confounds.columns.values,
                           regressors=confounds.T.values.tolist()) # movement regressors added here. List of lists.
        return output

    make_bunch = pe.Node(Function(input_names=['task_file', 'design_col', 'confounds', 'conditions'],
                                 output_names=['subject_info'],
                                  function=get_subj_info),
                         name='make_bunch')
    # make_bunch.inputs.task_file =  # From get_task
    # make_bunch.inputs.confounds =  # From get_confounds
    # make_bunch.inputs.design_col =  # From inputspec
    # make_bunch.inputs.conditions =  # From inputspec

    def mk_outdir(output_dir, options, proj_name):
        import os
        from time import gmtime, strftime
        prefix = proj_name
        if options['smooth']:
            new_out_dir = os.path.join(output_dir, prefix, 'smooth')
        else:
            new_out_dir = os.path.join(output_dir, prefix, 'nosmooth')
        if not os.path.isdir(new_out_dir):
            os.makedirs(new_out_dir)
        return new_out_dir

    make_outdir = pe.Node(Function(input_names=['output_dir', 'options', 'proj_name'],
                                   output_names=['new_out_dir'],
                                   function=mk_outdir),
                          name='make_outdir')
    # make_outdir.inputs.proj_name = from inputspec
    # make_outdir.inputs.output_dir = from inputspec
    make_outdir.inputs.options = options


    ################## Mask functional data.
    from jtnipyutil.util import mask_img
    maskBold = pe.Node(Function(input_names=['img_file', 'mask_file'],
                                output_names=['out_file'],
                                function=mask_img),
                      name='maskBold')
    # maskBold.inputs.img_file # From get_bold, or smooth_wf
    # maskBold.inputs.mask_file # From get_mask

    ################## Despike
    from nipype.interfaces.afni import Despike
    despike = pe.Node(Despike(),
                      name='despike')
    # despike.inputs.in_file = # From Mask
    despike.inputs.outputtype = 'NIFTI_GZ'

    from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
    smooth_wf = create_susan_smooth()
    # smooth_wf.inputs.inputnode.in_files = # from maskBold
    # smooth_wf.inputs.inputnode.fwhm = # from inputspec

    ################## Model Generation.
    import nipype.algorithms.modelgen as model
    specify_model = pe.Node(interface=model.SpecifyModel(), name='specify_model')
    specify_model.inputs.input_units = 'secs'
    # specify_model.functional_runs # From maskBold, despike, or smooth_wf
    # specify_model.subject_info # From make_bunch.outputs.subject_info
    # specify_model.high_pass_filter_cutoff # From inputspec
    # specify_model.time_repetition # From inputspec

    ################## Estimate workflow
    from nipype.workflows.fmri.fsl import estimate # fsl workflow
    modelfit = estimate.create_modelfit_workflow()
    modelfit.base_dir = '.'
    # modelfit.inputs.inputspec.session_info = # From specify_model
    # modelfit.inputs.inputspec.functional_data = # from maskBold
    # modelfit.inputs.inputspec.interscan_interval = # From inputspec
    # modelfit.inputs.inputspec.film_threshold = # From inputspec
    # modelfit.inputs.inputspec.bases = # From inputspec
    # modelfit.inputs.inputspec.model_serial_correlations = # From inputspec
    # modelfit.inputs.inputspec.contrasts = # From inputspec

    if not options['run_contrasts']: # drop contrast part of modelfit if contrasts aren't required.
        modelestimate = modelfit.get_node('modelestimate')
        merge_contrasts = modelfit.get_node('merge_contrasts')
        ztop = modelfit.get_node('ztop')
        outputspec = modelfit.get_node('outputspec')
        modelfit.disconnect([(modelestimate, merge_contrasts, [('zstats', 'in1'),
                                                             ('zfstats', 'in2')]),
                             (merge_contrasts, ztop, [('out', 'in_file')]),
                             (merge_contrasts, outputspec, [('out', 'zfiles')]),
                             (ztop, outputspec, [('out_file', 'pfiles')])
                            ])
        modelfit.remove_nodes([merge_contrasts, ztop])

    ################## DataSink
    from nipype.interfaces.io import DataSink
    import os.path
    sinker = pe.Node(DataSink(), name='sinker')
    # sinker.inputs.substitutions = # From inputspec
    # sinker.inputs.base_directory = # frm make_outdir

    def negate(input):
        return not input

    def unlist(input):
        return input[0]

    lvl1pipe_wf.connect([
        # grab subject/run info
        (inputspec, get_bold, [('subject_id', 'subj_id'),
                                ('bold_template', 'template')]),
        (inputspec, get_mask, [('subject_id', 'subj_id'),
                                ('mask_template', 'template')]),
        (inputspec, get_task, [('subject_id', 'subj_id'),
                                ('task_template', 'template')]),
        (inputspec, get_confile, [('subject_id', 'subj_id'),
                                ('confound_template', 'template')]),
        (inputspec, get_confounds, [('noise_transforms', 'noise_transforms'),
                                     ('noise_regressors', 'noise_regressors'),
                                     ('TR', 'TR')]),
        (inputspec, make_bunch, [('design_col', 'design_col'),
                                  ('conditions', 'conditions')]),
        (inputspec, make_outdir, [('output_dir', 'output_dir'),
                                  ('proj_name', 'proj_name')]),
        (inputspec, specify_model, [('hpf_cutoff', 'high_pass_filter_cutoff'),
                                     ('TR', 'time_repetition')]),
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
        lvl1pipe_wf.connect([
            (get_bold, despike, [('out_file', 'in_file')])
            ])
        if options['smooth']:
            lvl1pipe_wf.connect([
                (inputspec, smooth_wf, [('fwhm', 'inputnode.fwhm')]),
                (inputspec, get_gmmask, [('subject_id', 'subj_id'),
                                        ('smooth_gm_mask_template', 'template')]),
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
                (inputspec, get_gmmask, [('subject_id', 'subj_id'),
                                        ('smooth_gm_mask_template', 'template')]),
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
        (inputspec, sinker, [('subject_id','container'),
                              ('sinker_subs', 'substitutions')]), # creates folder for each subject.
        (make_outdir, sinker, [('new_out_dir', 'base_directory')]),
        (modelfit, sinker, [('outputspec.parameter_estimates', 'model'),
                            ('outputspec.dof_file','model.@dof'), #.@ puts this in the model folder.
                            ('outputspec.copes','model.@copes'),
                            ('outputspec.varcopes','model.@varcopes'),
                            ('outputspec.zfiles','stats'),
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
                            ('modelestimate.thresholdac', 'model.@serial_corr'),
                           ])
        ])
    if options['keep_resid']:
        lvl1pipe_wf.connect([
            (modelfit, sinker, [('modelestimate.residual4d', 'model.@resid')
                               ])
            ])
    return lvl1pipe_wf

# from jtnipyutil.util import combine_runs
import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface
import numpy as np
import pandas as pd
import os
import nibabel as nib
import glob
import math

#find the index of the closest number to K in a list
def closest(lst, K): 
    val = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
    idx = lst.index(val)
    return idx
#get consecutive intervals of 
def get_consecutive_intervals(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

#convert common event files to specific event files
# commonEventFile = '*_task-'+os.environ['PROJINIT']+'_run-*_events.tsv'
subj_dir = '/scratch/data/'+os.environ['SUBJ']+'/func/'

#get valid runs only
#run_list = pd.read_csv('/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/cleanup/EmoPain_good_run_all.csv', sep='\t', header=None)[0]
run_list = pd.read_csv('/scratch/wrkdir/EmoPain_good_run_all.csv', sep='\t', header=None)[0]
#run_list = list([i for i in run_list if 'sub-014' in i])
run_list = list([i for i in run_list if sys.argv[1] in i])
run_list = list([i for i in run_list if 'run' in i])
# run_list = run_list[1:2]
run_list.sort()
# run_list = glob.glob(os.path.join(subj_dir,commonEventFile))

run_modeling_criterion = pd.DataFrame()
for run_dir in run_list:
    run_id = run_dir
    run_dir = subj_dir+run_dir+'_events.tsv'
    subj_df_original = pd.read_table(run_dir, sep='\t')
    if 'pain' in run_dir:
        subj_ID = run_dir[-36:-29]
    elif 'emo' in run_dir:
        subj_ID = run_dir[-35:-28]
    run_num = run_dir[-13:-11]

    if len(glob.glob('/scratch/output/'+os.environ['PROJNAME']+'/nosmooth/'+subj_ID+'_task-'+os.environ['PROJINIT']+'_run-'+run_num+'/model/_*/_modelestimate0/cope1.nii.gz')) != 0:
        continue

    #event file to model CS and US  
    # subj_df_original = pd.read_table('/Users/chendanlei/Dropbox (Partners HealthCare)/U01/EmotionAvoidanceTask/BIDS_BehavioralData/sub-067/sub-067_task-pain3_run-04_events.tsv', sep='\t')
    # subj_df_original = pd.read_table('/Users/chendanlei/Dropbox (Partners HealthCare)/U01/EmotionAvoidanceTask/BIDS_BehavioralData/sub-070/sub-070_task-pain3_run-04_events.tsv', sep='\t')
    # subj_df_event = subj_df_original.copy()

    # for n in range(len(subj_df_event)):
    #     # print(n)
    #     if subj_df_event['trial_type'][n] == 'anticipation':
    #         #change anticipation row to a copy of US, and change the trial_type and duration
    #         subj_df_event.loc[n] = subj_df_event.loc[n+1].copy()
    #         subj_df_event.loc[n,'trial_type'] = 'US_onset'
    #         subj_df_event.loc[n,'duration'] = 1
    #         subj_df_event = subj_df_event.reset_index(drop=True)
    #         subj_df_event.loc[n+1,'trial_type'] = 'US_period'
    #         subj_df_event.loc[n+1,'onset'] = subj_df_event['onset'][n+1]+subj_df_event['duration'][n]
    #         subj_df_event.loc[n+1,'duration'] = 2
    #         subj_df_event = subj_df_event.reset_index(drop=True)
    subj_df_event = subj_df_original[(subj_df_original['trial_type'] == 'CS') | (subj_df_original['trial_type'] == 'US') | (subj_df_original['trial_type'] == 'decision')]
    subj_df_event = subj_df_event.reset_index(drop=True)
    
    #drop if onset column is nan
    subj_df_event = subj_df_event[subj_df_event['onset'].notna()]
    subj_df_event = subj_df_event.reset_index(drop=True)
    
    subj_df_event.loc[(subj_df_event['participant_made_choice']=='computer') & (subj_df_event['trial_type']=='decision'), 'trial_type'] = 'passive'
    for i in range(len(subj_df_event['RT'])):
        try:
            subj_df_event['RT'][i] = float(subj_df_event['RT'][i])
        except:
            subj_df_event['RT'][i] = np.nan
    subj_df_event.loc[(subj_df_event['participant_made_choice']=='participant') & (subj_df_event['trial_type']=='decision') & (pd.isnull(subj_df_event['RT'])), 'trial_type'] = 'active_no_motor'
    subj_df_event.loc[(subj_df_event['participant_made_choice']=='participant') & (subj_df_event['trial_type']=='decision') & (pd.notnull(subj_df_event['RT'])), 'trial_type'] = 'active_motor'
    # subj_df_event.loc[subj_df_event['US_trialType'] == 'negative','trial_type'] = ['USneg_'+i.split('_')[-1] for i in subj_df_event.loc[subj_df_event['US_trialType']=='negative','trial_type']]
    # subj_df_event.loc[subj_df_event['US_trialType'] == 'neutral','trial_type'] = ['USneu_'+i.split('_')[-1] for i in subj_df_event.loc[subj_df_event['US_trialType']=='neutral','trial_type']]
    # subj_df_event.loc[subj_df_event['CS_trialType'] == 'CS+','trial_type'] = 'CS+'
    # subj_df_event.loc[subj_df_event['CS_trialType'] == 'CS-','trial_type'] = 'CS-'

    subj_df_event=subj_df_event.astype(str)
    subj_df_event[subj_df_event=='nan']='NaN'
    subj_df_event = subj_df_event.reset_index(drop=True)
    # subj_df_event.to_csv(os.path.join('/Users/chendanlei/Desktop/','x.tsv'),index=True,sep='\t')

    event_file_name = os.environ['PROJINIT']+'_run-'+run_num+'_events_'+os.environ['PROJNAME']
    subj_df_event.to_csv(os.path.join(subj_dir,subj_ID+'_task-'+event_file_name+'.tsv'),index=True,sep='\t')

    if subj_ID == 'sub-070' and run_num=='04':
        print('caveat in '+subj_ID+'_'+run_num)
        subj_df_event = subj_df_event[0:40]
        
    # only include runs that meet certain modeling criterion to run in parallell
    if np.sum(subj_df_event['trial_type']=='active_no_motor')>0 and np.sum(subj_df_event['trial_type']=='active_motor')>0:
        run_modeling_criterion = run_modeling_criterion.append({'run':run_id,'criterion':'all'},ignore_index=True)

    elif np.sum(subj_df_event['trial_type']=='active_no_motor')==0 and np.sum(subj_df_event['trial_type']=='active_motor')>0:
        run_modeling_criterion = run_modeling_criterion.append({'run':run_id,'criterion':'all_motor'},ignore_index=True)

    elif np.sum(subj_df_event['trial_type']=='active_no_motor')>0 and np.sum(subj_df_event['trial_type']=='active_motor')==0:
        run_modeling_criterion = run_modeling_criterion.append({'run':run_id,'criterion':'all_no_motor'},ignore_index=True)

    #change the framewiseDisplacement data in the confound file to discount trials without onset using 'spike_regression'
    run = run_dir.split('-')[-1].split('_events.tsv')[0]
    print(run)
    # df_cf = pd.DataFrame(pd.read_csv('/Users/chendanlei/Desktop/sub-014_task-emo3_run-02_bold_confounds.tsv', sep='\t', parse_dates=False))
    # df_cf = pd.DataFrame(pd.read_csv('/Users/chendanlei/Desktop/sub-127_task-pain3_run-05_bold_confounds.tsv', sep='\t', parse_dates=False))
    df_cf = pd.DataFrame(pd.read_csv('/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-'+run+'_bold_confounds.tsv', sep='\t', parse_dates=False))
    fd = list(df_cf['FramewiseDisplacement'][:])
    TR=2.34
    num_TR = len(fd)
    TR_series = [x*TR for x in range(0, num_TR)]  

    nan_intervals = get_consecutive_intervals(list(subj_df_event['onset'].loc[subj_df_event['onset']=='NaN'].index))
    for nan_interval in nan_intervals:
        if nan_interval[0]==0:
            start_TR = 0
        else:
            start_TR = closest(TR_series, float(subj_df_event['onset'][nan_interval[0]-1]) + float(subj_df_event['duration'][nan_interval[0]-1]))
        if nan_interval[-1]==len(subj_df_event)-1:
            end_TR = len(TR_series)-1
        else:
            end_TR = closest(TR_series, float(subj_df_event['onset'][nan_interval[-1]+1]))
        #set the TR in between start and end TR to 100 (can be anything larger than 0.5 or the spike regression threshold)
        fd[start_TR:end_TR+1] = [100]*(end_TR+1-start_TR)
        
    df_cf['FramewiseDisplacement'] = fd
    df_cf.to_csv('/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-'+run+'_bold_confounds_temp.tsv',index=True,sep='\t')

print(run_modeling_criterion)
criterion_list = ['all', 'all_motor', 'all_no_motor']
for criterion in criterion_list:
    run_list_criterion = list(run_modeling_criterion['run'][run_modeling_criterion['criterion']==criterion])
    print(criterion)
    print(run_list_criterion)

    sinker_subs = [('subject_id_',''), # AROMA Setup
                    ('_fwhm','fwhm'),
                    ('subject_id_sub','sub'),
                    (' ','_'),]

    options = {'remove_steadystateoutlier': True,
               'smooth': False,
               'censoring': '',
               'ICA_AROMA': False,
               'poly_trend': 0, # intercept
               'dct_basis': 120,
              'run_contrasts': True,
              'keep_resid': True,
              'spike_regression': .5}

    model_wf = create_lvl1pipe_wf(options)
    model_wf.inputs.inputspec.input_dir = '/scratch/data/'
    model_wf.inputs.inputspec.output_dir = '/scratch/output/'
    model_wf.inputs.inputspec.design_col = 'trial_type'
    model_wf.inputs.inputspec.noise_regressors = ['X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*','aCompCorCSFnoFilter0','aCompCorCSFnoFilter1','aCompCorCSFnoFilter2','aCompCorCSFnoFilter3','aCompCorCSFnoFilter4','aCompCorWMnoFilter0','aCompCorWMnoFilter1','aCompCorWMnoFilter2','aCompCorWMnoFilter3','aCompCorWMnoFilter4']
    model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
    model_wf.inputs.inputspec.TR = 2.34  # In seconds, ensure this is a float
    model_wf.inputs.inputspec.FILM_threshold = 1 # Threshold for FILMGLS.  1000: p<=.001, 1: p <=1, i.e. unthresholded.
    model_wf.inputs.inputspec.hpf_cutoff = 0.
    if criterion == 'all':
        model_wf.inputs.inputspec.conditions = ['CS', 'US', 'passive', 'active_no_motor', 'active_motor', ['active_motor','RT', 'cent', 'norm']] # parameter to model from task file.
        model_wf.inputs.inputspec.contrasts = [['CS', 'T', ['CS'], [1]],      
                                               ['US', 'T', ['US'], [1]],         
                                               ['passive', 'T', ['passive'], [1]],                          
                                               ['active_no_motor', 'T', ['active_no_motor'], [1]],                          
                                               ['active_motor', 'T', ['active_motor'], [1]],
                                               ['active_motor_RT', 'T', ['active_motor_RT'], [1]],
                                               ['active_motor_no_motor_contrast', 'T', ['active_motor','active_no_motor'], [1,-1]],      
                                               ['active_motor_passive_contrast', 'T', ['active_motor','passive'], [1,-1]],                          
                                               ['active_no_motor_passive_contrast', 'T', ['active_no_motor','passive'], [1,-1]]] 
    elif criterion == 'all_motor':
        model_wf.inputs.inputspec.conditions = ['CS', 'US', 'passive', 'active_motor', ['active_motor','RT', 'cent', 'norm']] # parameter to model from task file.
        model_wf.inputs.inputspec.contrasts = [['CS', 'T', ['CS'], [1]],      
                                               ['US', 'T', ['US'], [1]],         
                                               ['passive', 'T', ['passive'], [1]],                          
                                               ['active_motor', 'T', ['active_motor'], [1]],
                                               ['active_motor_RT', 'T', ['active_motor_RT'], [1]],
                                               ['active_motor_passive_contrast', 'T', ['active_motor','passive'], [1,-1]]] 
    elif criterion == 'all_no_motor':
        model_wf.inputs.inputspec.conditions = ['CS', 'US', 'passive', 'active_no_motor'] # parameter to model from task file.
        model_wf.inputs.inputspec.contrasts = [['CS', 'T', ['CS'], [1]],      
                                               ['US', 'T', ['US'], [1]],         
                                               ['passive', 'T', ['passive'], [1]],                          
                                               ['active_no_motor', 'T', ['active_no_motor'], [1]],                          
                                               ['active_no_motor_passive_contrast', 'T', ['active_no_motor','passive'], [1,-1]]] 
    model_wf.inputs.inputspec.bases = {'dgamma':{'derivs': False}} # For more options, see Level1Design at https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.fsl/model.html
    model_wf.inputs.inputspec.model_serial_correlations = True # Include Pre-whitening, deals with autocorrelation
    model_wf.inputs.inputspec.sinker_subs = sinker_subs
    model_wf.inputs.inputspec.bold_template = {'bold': '/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'}
    model_wf.inputs.inputspec.mask_template = {'mask': '/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-*_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'}
    model_wf.inputs.inputspec.task_template = {'task': '/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-*_events_'+os.environ['PROJNAME']+'.tsv'}
    # model_wf.inputs.inputspec.confound_template = {'confound': '/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-'+run+'_bold_confounds_temp.tsv'} 
    model_wf.inputs.inputspec.confound_template = {'confound': '/scratch/data/'+sys.argv[1]+'/func/'+sys.argv[1]+'_task-'+os.environ['PROJINIT']+'_run-*_bold_confounds_temp.tsv'} 
    model_wf.inputs.inputspec.smooth_gm_mask_template = {'gm_mask': '/scratch/data/sub-*/anat/sub-*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz'}
    model_wf.inputs.inputspec.gmmask_args =  '-thr .5 -bin -kernel gauss 1 -dilM' # FSL Math command to adjust grey matter for susan smoothing mask.
    model_wf.inputs.inputspec.proj_name = os.environ['PROJNAME']

    # model_wf.inputspec.subject_id = # Could use 'sub-001', or use iterables below.
    # model_wf.inputspec.fwhm = # could use 1.5, or use iterables below.
    # subject_list = [i for i in run_list if 'run-'+run in i]
    subject_list = run_list_criterion
    fwhm_list = []

    infosource = pe.Node(IdentityInterface(fields=['fwhm', 'subject_id']),
               name='infosource')
    # infosource.iterables = [('fwhm', fwhm_list),
    #    ('subject_id', subject_list)]
    infosource.iterables = [('subject_id', subject_list)] # If no smoothing.

    full_model_wf = pe.Workflow(name='full_model_wf')
    # full_model_wf.connect([
    #     (infosource, model_wf, [('fwhm', 'inputspec.fwhm'),
    #                             ('subject_id', 'inputspec.subject_id')])])
    full_model_wf.connect([
        (infosource, model_wf, [('subject_id', 'inputspec.subject_id')])]) # If no smoothing.

    full_model_wf.base_dir = '/scratch/data/'+os.environ['PROJNAME']
    full_model_wf.crash_dump = '/scratch/data/crashdump'

    ########## Visualize ##################################################
    # full_model_wf.write_graph('simple.dot')
    # from IPython.display import Image
    # import os
    # Image(filename= os.path.join(full_model_wf.base_dir, 'full_model_wf','simple.png'))

    ########## RUN ##################################################
    full_model_wf.run(plugin='MultiProc', plugin_args={'n_procs': 2})






