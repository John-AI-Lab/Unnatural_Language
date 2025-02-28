conda activate hiddenlanguage

base_dir=./out
model=$1
model_id=$2
conv_template=$3
instruction_type=$4 # natural_instruction, unnatural_instruction, random_instruction, no_instruction

wandb offline

exp_name=${model_id}_${instruction_type}_lima
model_id=$exp_name
output_dir=$base_dir/${exp_name}
mkdir -p $output_dir
python train_lora.py \
  --data-path ../data/unnatural_lima/data.jsonl \
  --output_dir $output_dir \
  --wandb_run_name $exp_name \
  --base_model $model \
  --batch_size 32 \
  --micro_batch_size 2 \
  --learning_rate 0.0004 \
  --cutoff_len 4096 \
  --val_set_size 0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules '[gate_proj, down_proj, up_proj]' \
  --train_on_inputs False \
  --add_eos_token True \
  --group_by_length False \
  --lr_scheduler 'cosine' \
  --warmup_steps 100 \
  --wandb_project llm-attack \
  --num_epochs 10 \
  --conv_template $conv_template \
  --prompt_format instruction \
  --use_cot False \
  --instruction_type $instruction_type \
  --output_type output \
  2>&1 | tee $output_dir/log.txt

# mixeval inference
python -m mix_eval.evaluate \
  --model_name local_chat \
  --model_path ${output_dir} \
  --data_path ./MixEval/mix_eval/data \
  --benchmark mixeval \
  --version 2024-08-11 \
  --batch_size 16 \
  --output_dir ./data/bench/mixeval/out/${model_id} \
  --api_parallel_num 20 \
  --conv_template $conv_template \
  --inference_only

# alpaca_eval inference
python -m utils.eval.generate default \
  --bench alpaca_eval \
  --model_path $model \
  --adapter_model_path ${output_dir} \
  --conv_template $conv_template \
  --model_id $model_id \
  --batch_size 16 \
  --max_new_len 512 \
  --output_file_format json
