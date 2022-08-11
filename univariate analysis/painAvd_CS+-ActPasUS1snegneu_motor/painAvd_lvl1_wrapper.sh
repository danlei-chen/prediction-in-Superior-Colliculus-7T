#!/bin/tcsh
setenv DATA /autofs/cluster/iaslab/FSMAP/FSMAP_data
setenv SCRIPTPATH /autofs/cluster/iaslab/users/danlei/FSMAP/scripts
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/jtnipyutil/jtnipyutil-2019-01-03-4cecb89cb1d9.simg
setenv PROJNAME painAvd_CS+-ActPasUS1snegneu_motor
setenv PROJINIT pain3
setenv VALIDITYPATH /autofs/cluster/iaslab/users/danlei/validity/
setenv SINGULARITY /usr/bin/singularity
setenv SUBJ sub-$1

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/BIDS_preproc/$SUBJ/anat
mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/BIDS_preproc/$SUBJ/func
mkdir /scratch/$USER/$SUBJ/$PROJNAME/wrkdir/
mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/BIDS_modeled
mkdir /autofs/cluster/iaslab/FSMAP2/FSMAP_data/BIDS_modeled/$PROJNAME

rsync -ra $DATA/BIDS_fmriprep/fmriprep/ses-02/$SUBJ/func/*pain3_run-* /scratch/$USER/$SUBJ/$PROJNAME/BIDS_preproc/$SUBJ/func
rsync -ra $DATA/BIDS_fmriprep/fmriprep/ses-02/$SUBJ/anat/sub-*_T1w_space-MNI* /scratch/$USER/$SUBJ/$PROJNAME/BIDS_preproc/$SUBJ/anat
rsync -r $VALIDITYPATH /scratch/$USER/$SUBJ/$PROJNAME/wrkdir/

rsync $SCRIPTPATH/model/$PROJNAME/{painAvd_lvl1_model.py,painAvd_lvl1_model_startup.sh} /scratch/$USER/$SUBJ/$PROJNAME/wrkdir/
chmod +x /scratch/$USER/$SUBJ/$PROJNAME/wrkdir/painAvd_lvl1_model_startup.sh
cd /scratch/$USER
mkdir /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/$PROJNAME
chmod a+rwx /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/$PROJNAME

$SINGULARITY exec  \
--bind "/scratch/$USER/$SUBJ/$PROJNAME/BIDS_preproc:/scratch/data" \
--bind "/autofs/cluster/iaslab/FSMAP2/FSMAP_data/BIDS_modeled:/scratch/output" \
--bind "/scratch/$USER/$SUBJ/$PROJNAME/wrkdir:/scratch/wrkdir" \
$IMAGE\
/scratch/wrkdir/painAvd_lvl1_model_startup.sh

# rsync -r /scratch/$USER/$SUBJ/$PROJNAME/BIDS_modeled/ /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/
rm -r /scratch/$USER/$SUBJ/$PROJNAME/
exit

# scp -r /scratch/dz609/sub-127/painAvd_CSUS1snegneu/BIDS_preproc/painAvd_CSUS1snegneu/full_model_wf/lvl_one_pipe/modelfit/_subject_id_sub-127_task-pain3_run-05/modelgen/mapflow/_modelgen0 /autofs/cluster/iaslab/users/danlei/temp/
# rsync -r "dz609@door.nmr.mgh.harvard.edu:/autofs/cluster/iaslab/users/danlei/temp/_modelgen0" "/Users/chendanlei/Desktop/"
