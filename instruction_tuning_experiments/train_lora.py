import os
import random
import sys
from typing import List

import fire
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from fastchat.model import get_conversation_template
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                  set_peft_model_state_dict)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.prompt import Prompter


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = None,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "lora-moe",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: bool = True,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",
    expected_num_data: int = 10000,
    conv_template: str = "mistral",
    prompt_format: str = "instruction",  # instruction (default), no_instruction
    use_system_message: str = False,
    use_cot: bool = False,
    seed: int = 42,
    instruction_type="instruction",
    output_type="output",
    context_type=None,
    p_to_be_unnatural=0,
    merge_after_training: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt_format: {prompt_format}\n"
            f"p_to_be_unnatural: {p_to_be_unnatural}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    set_seed(seed)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        assert gradient_accumulation_steps > 0
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    is_llama = "llama" in base_model or "Llama" in base_model  # llama2 should use torch.bfloat16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if is_llama else torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    use_eager_kwargs = {}
    if "gemma" in base_model:
        use_eager_kwargs = {"attn_implementation": "eager"}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if is_llama else torch.float16,
        device_map=device_map,
        **use_eager_kwargs,
    )
    if is_llama:
        model.config.use_cache = False
        model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print(
        "pre-trained model's BOS EOS and PAD token id:",
        bos,
        eos,
        pad,
        " => It should be 1 2 None",
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def get_conv_template(conv_template, use_system_message=False):
        conv = get_conversation_template(conv_template)
        if conv.name == "zero_shot":
            conv.roles = tuple(["### " + r for r in conv.roles])
            conv.sep = "\n"
        elif conv.name == "llama-2":
            conv.sep2 = conv.sep2.strip()
        if not use_system_message:
            print("Not using system message")
            conv.system_message = ""
        return conv

    def apply_conv_template(conv, instruction, output=None):
        conv.messages = []
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], output)
        full_prompt = conv.get_prompt()
        return full_prompt

    # apply chat template and tokenize it
    conv = get_conv_template(conv_template, use_system_message)

    def generate_and_tokenize_prompt_with_dialogue(data_point, use_cot, instruction_type, output_type, context_type):
        def _apply_conv_template(conv, prompts):
            if not conv:
                return prompts
            if isinstance(prompts, str):
                prompts = [prompts]
            conv.messages = []
            # for i, prompt in enumerate(prompts):
            #     self.conv.append_message(self.conv.roles[i % 2], prompt)
            #     self.conv.append_message(self.conv.roles[i % 2], None)
            if len(prompts) % 2 == 1:
                prompts += [None]
            for i in range(len(prompts) // 2):
                conv.append_message(conv.roles[0], prompts[2 * i])
                conv.append_message(conv.roles[1], prompts[2 * i + 1])
            prompt = conv.get_prompt()
            prompt = prompt.replace("</s>", "")
            return prompt

        if use_cot:
            data_point[output_type] = "Let's think step by step. " + data_point[output_type]
        full_prompt = _apply_conv_template(
            conv, [data_point[context_type], "OK, got it", data_point[instruction_type], data_point[output_type]]
        )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = _apply_conv_template(
                conv, [data_point[context_type], "OK, got it", data_point[instruction_type]]
            )

            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_for_context_question(
        data_point, use_cot, instruction_type, output_type, context_type
    ):
        if use_cot:
            data_point[output_type] = "Let's think step by step. " + data_point[output_type]

        if context_type == "no_instruction":
            instruction = ""
        else:
            instruction = data_point[context_type] + "\n" + data_point[instruction_type]

        output = data_point[output_type]

        full_prompt = apply_conv_template(conv, instruction, output)

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = apply_conv_template(conv, instruction)

            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt

    def generate_and_tokenize_prompt(data_point, use_cot, instruction_type, output_type):
        if use_cot:
            data_point[output_type] = "Let's think step by step. " + data_point[output_type]

        instruction = data_point[instruction_type] if instruction_type != "no_instruction" else ""
        output = data_point[output_type]

        full_prompt = apply_conv_template(conv, instruction, output)

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = apply_conv_template(conv, instruction)

            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_with_no_instruction(data_point, use_cot, instruction_type, output_type):
        full_prompt = data_point["output"]
        try:
            output = tokenize(full_prompt)
            return output
        except Exception as e:
            __import__("ipdb").set_trace()

    def generate_and_tokenize_prompt_for_syncontextqa(data_point, use_cot=False):
        full_prompt = data_point["context"]
        output = tokenize(full_prompt)
        return output

    def generate_and_tokenize_prompt_with_p_to_be_unnatural(data_point, use_cot, output_type, p):
        if use_cot:
            data_point[output_type] = "Let's think step by step. " + data_point[output_type]

        if random.random() <= p:
            instruction_type = "instruction"
        else:
            instruction_type = "original_instruction"
        full_prompt = apply_conv_template(conv, data_point[instruction_type], data_point[output_type])

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = apply_conv_template(conv, data_point[instruction_type])

            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,  # increase learning ability
    )

    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if "debug" in wandb_run_name:
        data["train"] = data["train"].select(range(1000))

    # if resume_from_checkpoint:
    # Check the available weights and load them
    # checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
    # if not os.path.exists(checkpoint_name):
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "adapter_model.bin"
    #     )  # only LoRA model - LoRA config above has to fit
    #     resume_from_checkpoint = False  # So the trainer won't try loading its state
    # # The two files above have a different name depending on how they were saved, but are actually the same.
    # if os.path.exists(checkpoint_name):
    #     print(f"Restarting from {checkpoint_name}")
    #     adapters_weights = torch.load(checkpoint_name)
    #     set_peft_model_state_dict(model, adapters_weights)
    # else:
    #     print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if prompt_format == "no_instruction":
        tok_func = generate_and_tokenize_prompt_with_no_instruction
        tok_kwargs = {"use_cot": use_cot, "instruction_type": instruction_type, "output_type": output_type}
    elif prompt_format == "syncontextqa":
        tok_func = generate_and_tokenize_prompt_for_syncontextqa
        tok_kwargs = {"use_cot": use_cot, "instruction_type": instruction_type, "output_type": output_type}
    elif prompt_format == "p_to_be_unnatural":
        tok_func = generate_and_tokenize_prompt_with_p_to_be_unnatural
        tok_kwargs = {
            "use_cot": use_cot,
            "output_type": output_type,
            "p": p_to_be_unnatural,
        }
    elif prompt_format == "dialogue":
        tok_func = generate_and_tokenize_prompt_with_dialogue
        tok_kwargs = {
            "use_cot": use_cot,
            "instruction_type": instruction_type,
            "output_type": output_type,
            "context_type": context_type,
        }
    elif prompt_format == "context_question":
        tok_func = generate_and_tokenize_prompt_for_context_question
        tok_kwargs = {
            "use_cot": use_cot,
            "instruction_type": instruction_type,
            "output_type": output_type,
            "context_type": context_type,
        }
    else:
        tok_func = generate_and_tokenize_prompt
        tok_kwargs = {"use_cot": use_cot, "instruction_type": instruction_type, "output_type": output_type}
    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(lambda x: tok_func(x, **tok_kwargs))
        val_data = train_val["test"].shuffle().map(lambda x: tok_func(x, **tok_kwargs))
    else:
        train_data = data["train"].shuffle().map(lambda x: tok_func(x, **tok_kwargs))
        val_data = None

    # show example
    for i in range(5):
        text = tokenizer.decode(train_data[i]["input_ids"], skip_special_tokens=False)
        print(f"\nExample:\n{text}\n")

    if num_epochs is None and len(train_data) <= expected_num_data:  # use expected_num_data to compute num_epochs
        num_epochs = expected_num_data // len(train_data)
    print(f"num_epochs was set to {num_epochs}")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True if is_llama else False,
            fp16=True if not is_llama else False,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=100,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback], # ONLY USE LoadBestPeftModelCallback if val_set_size > 0
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # merge
    if merge_after_training:
        print("Merging LoRA weights with the base model...")
        model = model.merge_and_unload()

    # save lora
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(train)
