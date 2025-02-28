#!/bin/bash

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
conda activate hiddenlanguage

export n=25
model=mistralvicuna # llama2 or vicuna or vicuna_guanaco or 7b-models or vicuna_llama2
target_type=natural_context
goal_type=builtin
goal_pos=suffix

allow_non_ascii=False
use_empty_goal=False
word_blacklist_type=none
word_blacklist_topk=0
instruction_init=original_perturb # original, independent_sample, placeholder

start_offset=$1
end_offset=$2

batch_size=256

today=$(date '+%m-%d')
echo $today
exp_name=unnatural_simgsm8k
output_dir=../out/$exp_name
mkdir -p $output_dir

# check if a last completed offset file exists and read from it.
# resume_file="${output_dir}/resume_start_offset${start_offset}_end_offset${end_offset}.log"
# if [ -f "$resume_file" ]; then
#   last_completed_offset=$(<"$resume_file")
#   start_offset=$((last_completed_offset + 1))
# fi

for ((offset = $start_offset; offset < $end_offset; offset++)); do
  save_path=$output_dir/offset${offset} # replace space with underscore
  mkdir -p $save_path
  python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=reg_gcg \
    --config.target_type=$target_type \
    --config.goal_type=$goal_type \
    --config.goal_pos=$goal_pos \
    --config.word_blacklist_type=$word_blacklist_type \
    --config.word_blacklist_topk=$word_blacklist_topk \
    --config.use_empty_goal=$use_empty_goal \
    --config.train_data=vermouthdky/Unnatural_SimGSM8K \
    --config.result_prefix="$save_path/results" \
    --config.progressive_goals=False \
    --config.progressive_models=False \
    --config.allow_non_ascii=$allow_non_ascii \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.data_offset=$offset \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=$batch_size \
    --config.lr=0.01 \
    --config.max_train_words=32 \
    --config.filter_cand=True \
    --config.stop_on_success=False \
    --config.early_stop_patience=500 \
    --config.instruction_init=$instruction_init \
    --config.control_length=40 \
    --config.anneal=True \
    2>&1 | tee $save_path/log.txt
  # update last completed offset after successful completion
  echo $offset >${resume_file}
done

# collect and evaluate
# base_dir=$output_dir
# file_paths=""
# for ((offset=1; offset<${#concept_list[@]}; offset++)); do
#     file_paths=$file_paths,$base_dir/offset${offset}_${concept_list[$offset]}/log.txt
# done
# python -m experiments.utils collect_best_control --file_paths $file_paths --save_path $base_dir
# python -m experiments.utils generate --data_path $base_dir
# python -m experiments.utils evaluate --data_path $base_dir --eval_type heuristic
