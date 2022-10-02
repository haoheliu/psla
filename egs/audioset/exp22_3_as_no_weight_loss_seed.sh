# If we remove the threshold of the zero loss, what would happen?

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
  lrscheduler_start=35
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
hop_ms=10
target_length=1056
batch_size=24
learn_pos_emb=False
lambda_zero_loss=0.001
apply_zero_loss_threshold=0.0
weight_func=calculate_class_weight_min
graph_weight_path=undirected_graph_connectivity_no_root.npy
note=exp22_no_rw

for preserve_ratio in 1.0
do
  date=$(date '+%Y-%m-%d_%H_%M_%S')
  alpha=1.0
  beta=1.0
  
  reweight_loss=False
  full_bal=True
  
  seed=$1
  sampler=DoNothing

  echo $sampler
  exp_dir=./exp/${date}-${dataset}-${sampler}-posemb${learn_pos_emb}-zl${lambda_zero_loss}-${apply_zero_loss_threshold}-alpha${alpha}-beta${beta}-hop${hop_ms}-${target_length}-${reweight_loss}-${preserve_ratio}-${model}-${eff_b}-${lr}-${subset}-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-seed${seed}
  mkdir -p $exp_dir
  python ../../src/run.py --data-train $trpath --data-val ./data/datafiles/audioset_eval_data.json \
  --exp-dir $exp_dir --n-print-steps 50 --save_model True --num-workers 8 --label-csv ./data/class_labels_indices.csv \
  --n_class 527 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
  --model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 --hop_ms ${hop_ms} --apply_zero_loss_threshold ${apply_zero_loss_threshold} \
  --dataset_mean -7.4106 --dataset_std 6.3097 --target_length ${target_length} --noise False --lambda_zero_loss ${lambda_zero_loss} \
  --metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 --learn_pos_emb ${learn_pos_emb} \
  --wa True --wa_start ${wa_start} --wa_end ${wa_end} --sampler ${sampler} --preserve_ratio ${preserve_ratio} --val_interval 1 --note ${note} \
  --alpha ${alpha} --beta ${beta} --graph_weight_path ${graph_weight_path} --reweight_loss ${reweight_loss} --weight_func ${weight_func} --dataset ${dataset} --seed ${seed}
done
