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
date=$(date '+%Y-%m-%d_%H_%M_%S')
att_head=4
model=efficientnet
psla=True
eff_b=2
dataset=fsd50k

if [ "$psla" = "True" ]
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

for preserve_ratio in 0.1 0.25 0.5 0.05
do
alpha=1.0
beta=1.0
epoch=40
wa_start=21
wa_end=40
lrscheduler_start=10
hop_ms=10
batch_size=8
graph_weight_path=undirected_graph_connectivity_no_root.npy

note=exp10_10ms_fsd50k

learn_pos_emb=False
target_length=3000
reweight_loss=False
seed=21195
weight_func=calculate_class_weight_v10
sampler=$1
lambda_zero_loss=1.0

exp_dir=./exp/${date}-${dataset}-${sampler}-posemb${learn_pos_emb}-alpha${alpha}-beta${beta}-hop${hop_ms}-${target_length}-${reweight_loss}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-seed${seed}
# exp_dir=./exp/avg-pool-0.1-${model}-${eff_b}-${lr}-fsd50k-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-le${p}-2
mkdir -p $exp_dir

python ../../src/run.py --data-train $trpath --data-val ./datafiles/fsd50k_val_full.json --data-eval ./datafiles/fsd50k_eval_full.json --dataset ${dataset} \
--exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 8 --label-csv ./class_labels_indices.csv \
--n_class 200 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
--model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} --hop_ms ${hop_ms} --seed ${seed} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 --reweight_loss ${reweight_loss} --weight_func ${weight_func} \
--dataset_mean -13.9325 --dataset_std 3.7020 --target_length ${target_length} --noise False --learn_pos_emb ${learn_pos_emb} --lambda_zero_loss ${lambda_zero_loss} \
--metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 --note ${note} \
--alpha ${alpha} --beta ${beta} --graph_weight_path ${graph_weight_path} --wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1
done