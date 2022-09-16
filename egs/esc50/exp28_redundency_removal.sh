#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

conda activate psla
export TORCH_HOME=./
att_head=4
model=efficientnet
psla=True
eff_b=2

dataset=esc50

if [ "$psla" = "True" ]
then
  impretrain=True
  freqm=48
  timem=96
  mixup=0.5
  bal=False
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  bal=False
fi

batch_size=48
lr=2.5e-4
epoch=80 
wa_start=80
wa_end=80
lrscheduler_start=40 
target_length=500

lambda_zero_loss=1.0
apply_zero_loss_threshold=0.5

weight_func=calculate_class_weight_min
graph_weight_path=undirected_graph_connectivity_no_root.npy
alpha=1.0
beta=0.0
reweight_loss=False

note=exp28_esc
learn_pos_emb=False 

sampler=$1
preserve_ratio=$2
hop_ms=10

seed=21195

date=$(date '+%Y-%m-%d_%H_%M_%S')
base_exp_dir=./exp/${date}-${dataset}-${sampler}-zl${apply_zero_loss_threshold}${lambda_zero_loss}-posemb${learn_pos_emb}-alpha${alpha}-beta${beta}-hop${hop_ms}-${target_length}-${reweight_loss}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-seed${seed}
mkdir -p $base_exp_dir

for fold in 1 2 3 4 5
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}
  mkdir -p $exp_dir

  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json
  
  python ../../src/run.py --data-train ${tr_data} --data-val ${te_data} --data-eval ${te_data} \
  --exp-dir $exp_dir --n-print-steps 10 --save_model False --num-workers 8 --label-csv ./data/esc_class_labels_indices.csv \
  --n_class 50 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr --dataset ${dataset} \
  --model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} --hop_ms ${hop_ms} --seed ${seed} --apply_zero_loss_threshold ${apply_zero_loss_threshold} \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 --reweight_loss ${reweight_loss} --weight_func ${weight_func} \
  --dataset_mean -9.02 --dataset_std 6.07 --target_length ${target_length} --noise False --learn_pos_emb ${learn_pos_emb} --lambda_zero_loss ${lambda_zero_loss} \
  --metrics acc --warmup False --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.95 --note ${note} \
  --alpha ${alpha} --beta ${beta} --graph_weight_path ${graph_weight_path} --wa False --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1

done

python ./get_esc_result.py --exp_path ${base_exp_dir}
