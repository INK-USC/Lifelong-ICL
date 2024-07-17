#!/bin/bash

cd ../../

MODEL_NAME="" # your model name
SPLIT_NAME="default"

N_TASK=64
N_SHOT=2


SETTING_NAME="replay"
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 

SETTING_NAME="repeat"
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
