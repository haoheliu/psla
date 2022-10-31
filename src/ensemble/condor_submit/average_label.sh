eval "$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate psla
which python

######################## ENVIRONMENT ########################
cd /mnt/fast/nobackup/users/hl01486/hhl_script2/2022/segment_level_audio_tagging/psla/src/ensemble
######################## SETUP ########################

python3 average_labels.py