#!/bin/bash
set -euxo pipefail

export WANDB_MODE=online

export WANDB_BASE_URL=http://100.96.31.198:20001
export WANDB_API_KEY=local-xxxx

wandb login --relogin --host $WANDB_BASE_URL $WANDB_API_KEY

export HF_HUB_OFFLINE=1

cd BrainJanus/src

rsync -av --progress \
    --include='/voxel/' \
    --include='/voxel/subj0[1-8]_test_avg_bf16.safetensors' \
    --include='/voxel/subj0[1-8]_train_bf16.safetensors' \
    --exclude='/voxel/*' \
    --include='/image/' \
    --include='/image/nsd_stimuli_384_uint8.safetensors' \
    --exclude='/image/*' \
    --include='/captions/' \
    --include='/captions/*' \
    --exclude='*' \
    /data/ /dev/shm/data/

CKPT="checkpoints/Pretrain_fMRI_xxxx/epoch_xxx"

time_str=$(date +"%Y%m%d_%H%M")
name="Finetune_${time_str}"

echo $name
train_tag="_train_avg_bf16"
test_tag="_test_avg_bf16"

#single task
accelerate launch --config_file=config/accelerate_configs/zero2.yaml --num_processes 8 train_final.py --task 0 --name $name --subj_list "[1]" --epoch 30 --batch_size=32 --test_batch_size=64 --train_tag $train_tag --test_tag $test_tag
# accelerate launch --config_file=config/accelerate_configs/zero2.yaml --num_processes 8 train_final.py --task 1 --name $name --subj_list "[1,2,3,4,5,6,7,8]" --epoch 30 --batch_size=32 --test_batch_size=64 --train_tag $train_tag --test_tag $test_tag
# accelerate launch --config_file=config/accelerate_configs/zero2.yaml --num_processes 8 train_final.py --task 2 --name $name --subj_list "[1,2,5,7]" --epoch 15 --batch_size=32 --test_batch_size=64 --train_tag $train_tag --test_tag $test_tag
# accelerate launch --config_file=config/accelerate_configs/zero2.yaml --num_processes 8 train_final.py --task 3 --name $name --subj_list "[1]" --epoch 15 --batch_size=32 --test_batch_size=64 --train_tag $train_tag --test_tag $test_tag

#mix tasks
accelerate launch --config_file=config/accelerate_configs/zero2.yaml --num_processes 8 train_final.py --name $name --subj_list "[1]" --epoch 30 --batch_size=32 --test_batch_size=64 --train_tag $train_tag --test_tag $test_tag

python plot/compute_plot_metric.py --name $name
