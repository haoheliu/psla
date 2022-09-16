#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./log_%j.txt

conda activate psla
export TORCH_HOME=./
att_head=4
model=efficientnet
psla=True
eff_b=2

if [ "$psla" = "True" ]
then
  impretrain=True
  freqm=48
  timem=76
  mixup=0.5
  bal=True
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  bal=False
fi

lr=1e-4

lrscheduler_start=10
epoch=20
wa_start=11
wa_end=20

dataset=nsynth_pitch
lr=1e-4
hop_ms=10
lrscheduler_start=10
epoch=20
wa_start=11
wa_end=20
target_length=400
batch_size=128

lambda_zero_loss=1.0
apply_zero_loss_threshold=0.5

weight_func=calculate_class_weight_min
graph_weight_path=undirected_graph_connectivity_no_root.npy
alpha=1.0
beta=0.0
reweight_loss=False

note=exp27_nsynth_pitch_ultra
learn_pos_emb=False
seed=21195

sampler=$1
preserve_ratio=$2

tr_data=./data/datafiles/nsynth_pitch_train.json
val_data=./data/datafiles/nsynth_pitch_valid.json
eval_data=./data/datafiles/nsynth_pitch_test.json

date=$(date '+%Y-%m-%d_%H_%M_%S')
exp_dir=./exp/${date}-${dataset}-${sampler}-posemb${learn_pos_emb}-alpha${alpha}-beta${beta}-hop${hop_ms}-${target_length}-${reweight_loss}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-seed${seed}
mkdir -p $exp_dir

python ../../src/run.py --data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} \
--exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 8 --label-csv ./data/nsynth_pitch_class_labels_indices.csv \
--n_class 128 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr --dataset ${dataset} \
--model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} --hop_ms ${hop_ms} --seed ${seed} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 --reweight_loss ${reweight_loss} --weight_func ${weight_func} \
--dataset_mean -11.71 --dataset_std 5.38 --target_length ${target_length} --noise False --learn_pos_emb ${learn_pos_emb} --lambda_zero_loss ${lambda_zero_loss} \
--metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.85 --note ${note} \
--alpha ${alpha} --beta ${beta} --graph_weight_path ${graph_weight_path} --wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1
