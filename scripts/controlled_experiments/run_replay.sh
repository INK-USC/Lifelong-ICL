#!/bin/bash

cd ../../

MODEL_NAME="" # set model name
SPLIT_NAME="default"
SETTING_NAME="replay"

N_TASK=16
N_SHOT=4

# run replay

python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 