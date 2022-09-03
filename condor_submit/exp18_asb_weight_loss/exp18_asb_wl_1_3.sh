######################## ENVIRONMENT ########################
eval "$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate psla
which python

######################## SETUP ########################
DATA="/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav&/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav"
LOG="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER/egs"
PROJECT="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER"

######################## RUNNING ENTRY ########################
cd /mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER/egs/audioset

sh exp18_asb_wl.sh 1.3
sh exp19_asb_wl_no_bce.sh 1.3