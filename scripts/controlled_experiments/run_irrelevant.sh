#!/bin/bash

cd ../../

MODEL_NAME="" # set model name
SPLIT_NAME="default"
SETTING_NAME="repeat"

N_TASK=16
N_SHOT=4

for N_REPEAT in 2 4 8 16 32
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT}_irr${N_REPEAT} \
--n_repeat ${N_REPEAT} --use_irrlevant_prefix \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done
