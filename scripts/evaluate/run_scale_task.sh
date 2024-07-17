#!/bin/bash

cd ../../

# Run scaling tasks experiments

MODEL_NAME="" # your model name
SPLIT_NAME="default"
SETTING_NAME="recall"
N_SHOT=2


for N_TASK in 8 24 32 40 48 56 64
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done