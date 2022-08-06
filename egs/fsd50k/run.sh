#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2,9]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="psla_fsd"
#SBATCH --output=./log_%j.txt

# set -x
conda activate psla
export TORCH_HOME=./
date=$(date '+%Y-%m-%d-%H-%M')
att_head=4
model=efficientnet
psla=True
eff_b=7
batch_size=8
dataset=fsd50k
# pooling=False

# if [ $pooling == True ]
# then
#   drop_rate=0.1
#   batch_size=96
# else
#   drop_rate=0.1
#   batch_size=96
# fi

if [ $psla == True ]
then
  impretrain=True
  freqm=48
  timem=192
  mixup=0.5
  bal=True
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  bal=False
fi

lr=5e-4

trpath=./datafiles/fsd50k_tr_full.json

sampler=NoAction
preserve_ratio=0.5

epoch=40
wa_start=21
wa_end=40
lrscheduler_start=10

for sampler in NeuralSamplerPosEmbLearnableLargeEnergy NeuralSamplerPosEmbLearnableLargeEnergyNN NeuralSamplerPosEmbLearnableLargeEnergy 
do
exp_dir=./exp/${date}-${dataset}-${sampler}-${preserve_ratio}-${model}-${eff_b}-${lr}-fsd50k-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-le${p}-2
# exp_dir=./exp/avg-pool-0.1-${model}-${eff_b}-${lr}-fsd50k-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-le${p}-2
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0 python ../../src/run.py --data-train $trpath --data-val ./datafiles/fsd50k_val_full.json --data-eval ./datafiles/fsd50k_eval_full.json \
--exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 16 --label-csv ./class_labels_indices.csv \
--n_class 200 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
--model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
--dataset_mean -4.6476 --dataset_std 4.5699 --target_length 3000 --noise False \
--metrics mAP --warmup False --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
--wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1
done
