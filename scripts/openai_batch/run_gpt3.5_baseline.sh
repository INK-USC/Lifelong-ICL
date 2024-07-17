cd ../..

model="gpt-3.5"
n_task=4
split="default"
n_split_perm=1
permutation_num=0

for n_shot in  0 1 ; do
    output_dir="output/openai/baseline_${model}_${split}_${n_task}_${n_shot}_${permutation_num}"
    python prepare_openai_baseline.py --model $model --split $split \
        --n_shot $n_shot --n_task $n_task --output_dir $output_dir \
        --permutation_num $permutation_num --n_split_perm $n_split_perm

    python run_openai.py --split $split --result_dir $output_dir --model $model --keep_waiting
done