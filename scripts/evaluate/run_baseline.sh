#!/bin/bash

cd ../../

# Run baseline experiments

MODEL_NAME="" # your model name
SPLIT_NAME="default"
SETTING_NAME="baseline"

python cli.py \
    --output_dir output/${SPLIT_NAME}/${SETTING_NAME}/${MODEL_NAME}/ \
    --model ${MODEL_NAME} --exp_mode ${SETTING_NAME} --split ${SPLIT_NAME} 

