#!/bin/bash

# GPUs to be used
export CUDA_VISIBLE_DEVICES=0,1

# setup model
MODEL=resnet56
BATCH_SIZE=128
NUM_EPOCHS=182
DATA_DIR="$(pwd)/data"

LOG_DIR="--log_dir $(pwd)/checkpoints/${MODEL}_${BATCH_SIZE}"
SAVE_BEST="--save_best"
SAVE_EVERY=  #"--save_every 20"
NUM_CKPTS="--num_ckpts 10"
RESUME=  #"--resume"
PROGRESS_BAR=  #"--progress_bar"
LOG_OPTIONS="${LOG_DIR} ${SAVE_BEST} ${SAVE_EVERY} \
             ${NUM_CKPTS} ${RESUME} ${PROGRESS_BAR}"

LR="0.1"
DECAY_POLICY="pconst"
LR_MILESTONES="91 136"

# output logs
OUTPUT_DIR="$(pwd)/output"
OUTPUT_FILE=${MODEL}_${BATCH_SIZE}.out
mkdir -p $OUTPUT_DIR

# run 
SCRIPT=main.py

python $SCRIPT \
  --model ${MODEL} \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --data_dir ${DATA_DIR} \
  ${LOG_OPTIONS} \
  --lr ${LR} \
  --lr_decay_policy ${DECAY_POLICY} \
  --lr_milestones ${LR_MILESTONES} \
> $OUTPUT_DIR/$OUTPUT_FILE

