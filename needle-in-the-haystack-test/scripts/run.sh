#!/bin/bash

cd ../../

models=("film-7b")
haystack_dir="needle-in-the-haystack-test/data/PaulGrahamEssays"
output_dir="needle-in-the-haystack-test/output/"
prompt_dir="needle-in-the-haystack-test/prompts/"

min_length=1000
max_length=32000
depth_pcnt_n=16
test_length_n=16

for model in "${models[@]}"; do 
    python -m needle-in-the-haystack-test.run --model $model \
        --haystack_dir $haystack_dir \
        --output_dir $output_dir --prompt_dir $prompt_dir \
        --max_length $max_length --min_length $min_length --depth_pcnt_n $depth_pcnt_n \
        --test_length_n $test_length_n --all
done