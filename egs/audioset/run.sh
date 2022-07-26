#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2,9]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="psla_as"
#SBATCH --output=./log_%j.txt

conda activate psla
export TORCH_HOME=./

subset=balanced
att_head=4
model=efficientnet
psla=True
eff_b=2
date=$(date '+%Y-%m-%d-%H-%M')

if [ $psla == True ]
then
  impretrain=True
  freqm=48
  timem=192
  mixup=0.5
  full_bal=True
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  full_bal=False
fi

if [ $subset == balanced ]
then
  bal=False
  lr=1e-3
  p=mean
  # label enhanced set
  trpath=/media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/datafiles/audioset_bal_train_data.json
  # original set
  #trpath=./datafiles/balanced_train_data.json
  epoch=100
  wa_start=41
  wa_end=100
  lrscheduler_start=35
else
  bal=${full_bal}
  lr=1e-4
  p=None
  #trpath=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data_type1_2_${p}.json
  trpath=/media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/datafiles/audioset_bal_unbal_train_data.json
  epoch=30
  wa_start=16
  wa_end=30
  lrscheduler_start=10
fi

# echo $sampler
# exp_dir=./exp/${date}-${sampler}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-git
# mkdir -p $exp_dir

# CUDA_VISIBLE_DEVICES=0 python ../../src/run.py --data-train $trpath --data-val /media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/datafiles/audioset_eval_data.json \
# --exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 16 --label-csv /media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/class_labels_indices.csv \
# --n_class 527 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
# --model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
# --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
# --dataset_mean -4.6476 --dataset_std 4.5699 --target_length 1056 --noise False \
# --metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
# --wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} 

preserve_ratio=0.1
batch_size=96

#############################
  # freqm=0
  # timem=0
#############################

for sampler in NeuralSamplerPosEmbLearnableLargeEnergyNN NeuralSamplerPosEmbLearnableLargeEnergy NeuralSamplerAvgPool NeuralSamplerPosEmbLearnableLargeEnergyNN NeuralSamplerPosEmbLearnableLargeEnergy NeuralSamplerAvgPool
do
echo $sampler
exp_dir=./exp/${date}-${sampler}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-git-{$RANDOM}
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0 python ../../src/run.py --data-train $trpath --data-val /media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/datafiles/audioset_eval_data.json \
--exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 16 --label-csv /media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/class_labels_indices.csv \
--n_class 527 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
--model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
--dataset_mean -4.6476 --dataset_std 4.5699 --target_length 1056 --noise False \
--metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
--wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio}  --score_loss True --val_interval 10
# echo $sampler
done