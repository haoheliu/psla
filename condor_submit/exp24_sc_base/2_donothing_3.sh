######################## ENVIRONMENT ########################
eval "$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate psla
which python

######################## SETUP ########################
DATA="/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav&/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav"
LOG="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER/egs"
PROJECT="/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER"

######################## RUNNING ENTRY ########################
cd /mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/psla_$IDENTIFIER/egs/speechcommands

sh run_sc.sh $1 3.3333 3