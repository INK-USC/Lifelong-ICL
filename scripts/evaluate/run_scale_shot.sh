#!/bin/bash

cd ../../

# Run scaling shot experiments

MODEL_NAME="" # your model name
SPLIT_NAME="default"
SETTING_NAME="recall"
N_TASK=16


for N_SHOT in 1 2 3 4 5 6 7 8
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done

