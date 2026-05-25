#!/bin/bash
set -euxo pipefail

export WANDB_MODE=online

export WANDB_BASE_URL=http://100.96.31.198:20001
export WANDB_API_KEY=local-6e9bac02f4243b8ba11c6525317e5b7723a20f2b

wandb login --relogin --host $WANDB_BASE_URL $WANDB_API_KEY

export HF_HUB_OFFLINE=1

cd BrainRosetta/src


NUM_PROCESSES=8
CONFIG_FILE="config/accelerate_configs/zero2.yaml"
TRAIN_TAG="_train_bf16"
TEST_TAG="_test_avg_bf16"

# mkdir -p /dev/shm/data
rsync -av --progress \
    --exclude='/plot/' \
    --include='/voxel/' \
    --include='/voxel/subj0[1-8]_test_avg_bf16.safetensors' \
    --include='/voxel/subj0[1-8]_train_bf16.safetensors' \
    --exclude='/voxel/*' \
    /data /dev/shm/data

#---------------------Pretrain---------------------
# CODEBOOK_SIZES=(64 128 256 512 1024)
CODEBOOK_SIZES=(256)
for size in "${CODEBOOK_SIZES[@]}"
do

    NAME="Pretrain_fMRI_codebook${size}"
    
    echo "========================================================================"
    echo "Start Training: $NAME | Codebook Size: $size"
    echo "Start Time: $(date)"
    echo "========================================================================"

    accelerate launch --config_file="$CONFIG_FILE" --num_processes "$NUM_PROCESSES" \
        train_fmri_vq_vae.py \
        --name "$NAME" \
        --codebook_size="$size" \
        --epoch 5 \
        --subj_list "[1,2,3,4,5,6,7,8]" \
        --batch_size 256 \
        --test_batch_size 512 \
        --train_tag "$TRAIN_TAG" \
        --test_tag "$TEST_TAG"

    echo "$NAME FINISHED！"
    echo ""
done