#!/bin/bash

cd ../../

MODEL_NAME="" # set model name
SPLIT_NAME="default"

N_TASK=16
N_SHOT=4

SETTING_NAME="repeat"

# run repeat

for N_REPEAT in 2 4 8 16 32 64
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT}_repeat${N_REPEAT} \
--n_repeat ${N_REPEAT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done

# run repeat with shuffle

for N_REPEAT in 2 4 8 16 32 64
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT}_repeat${N_REPEAT}_shuffled \
--n_repeat ${N_REPEAT} --repeat_shuffle \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done
