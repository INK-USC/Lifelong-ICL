#!/bin/bash

cd ../../

MODEL_NAME="" # set model name
SPLIT_NAME="default"
SETTING_NAME="recall"

# Run scaling shot experiments

N_TASK=16

for N_SHOT in 1 2 3 4 5 6 7 8
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done

# Run scaling tasks experiments

N_SHOT=2

for N_TASK in 8 24 32 40 48 56 64
do
python cli.py \
--output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ntask${N_TASK}_nshot${N_SHOT} \
--model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --n_task ${N_TASK} --n_shot ${N_SHOT} --split ${SPLIT_NAME} 
done

