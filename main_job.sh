#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=0-24:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

module load python/3.7.4
module load cuda/11.4.0
pushd /home/gaotian/qagnn
source env/bin/activate



export CUDA_VISIBLE_DEVICES=0,1
dt=`date '+%Y%m%d_%H%M%S'`


dataset="csqa"
model='albert-xxlarge-v2'
shift
shift
args=$@


elr="1e-5"
dlr="1e-3"
bs=64
mbs=2
n_epochs=10
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=4 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 0; do
  python3 -u qagnn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.pr.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.pr.pk \
      --test_adj  data/${dataset}/graph/test.graph.pr.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done