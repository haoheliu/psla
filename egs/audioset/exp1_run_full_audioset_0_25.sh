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

subset=full
att_head=4
model=efficientnet
psla=True
eff_b=2

if [ "$psla" = "True" ]
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

if [ "$subset" = "balanced" ]
then
  dataset=audiosetbalanced
  bal=False
  lr=1e-3
  p=mean
  # label enhanced set
  trpath=./data/datafiles/audioset_bal_train_data.json
  # original set
  #trpath=./datafiles/balanced_train_data.json
  epoch=80
  wa_start=41
  wa_end=80 
  lrscheduler_start=40
elif [ "$subset" = "full" ]
then
  dataset=audioset
  bal=${full_bal}
  lr=1e-4
  p=None
  #trpath=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data_type1_2_${p}.json
  trpath=./data/datafiles/audioset_bal_unbal_train_data.json
  epoch=30
  wa_start=16
  wa_end=30
  lrscheduler_start=10
else
  echo "Error: Illegal subset name."
  exit 1
fi

#############################################
# 0.1 0.05 0.025 
batch_size=24

for preserve_ratio in 0.25
do
  note=as
  echo $preserve_ratio
  alpha=1.0
  beta=1.0
  graph_weight_path=undirected_graph_connectivity_no_root.npy
  reweight_loss=False
  full_bal=True
  weight_func=calculate_class_weight_v10
  seed=21195
  sampler=$1

  echo $sampler
  exp_dir=./exp/${dataset}-${sampler}-alpha${alpha}-beta${beta}-${reweight_loss}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-seed${seed}
  mkdir -p $exp_dir
  python ../../src/run.py --data-train $trpath --data-val ./data/datafiles/audioset_eval_data.json \
  --exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 8 --label-csv ./data/class_labels_indices.csv \
  --n_class 527 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
  --model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
  --dataset_mean -4.6476 --dataset_std 4.5699 --target_length 1056 --noise False \
  --metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
  --wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1 --note ${note} \
  --alpha ${alpha} --beta ${beta} --graph_weight_path ${graph_weight_path} --reweight_loss ${reweight_loss} --weight_func ${weight_func} --dataset ${dataset} --seed ${seed}
done
