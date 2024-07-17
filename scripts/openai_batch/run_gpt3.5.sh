cd ../..

model="gpt-3.5"
n_shot=2
n_task=64
split="default"
n_split_perm=1
permutation_num=1
output_dir="output/openai/${model}_${split}_${n_task}_${n_shot}_${permutation_num}"

python prepare_openai.py --model $model --split $split \
    --n_shot $n_shot --n_task $n_task --output_dir $output_dir --permutation_num $permutation_num --n_split_perm $n_split_perm

python run_openai.py --split $split --result_dir $output_dir --model $model --keep_waiting