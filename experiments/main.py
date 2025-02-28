"""A main script to run attack for LLMs."""

import importlib
import random
import time
from random import randint

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags

from unnatural_language import (get_goals_and_targets,
                                get_goals_and_targets_jsonl, get_workers)

_CONFIG = config_flags.DEFINE_config_file("config")


def inject_exclamations(input_string, target_length):
    words = input_string.split()
    while len(words) < target_length:
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, "!")
    return " ".join(words)


# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)


def main(_):
    mp.set_start_method("spawn")

    params = _CONFIG.value

    attack_lib = dynamic_import(f"unnatural_language.{params.attack}")

    print(params)

    if params.train_data.endswith(".csv"):
        train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)
        process_fn = lambda s: s.replace("Sure, h", "H")
        process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
        train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
        test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]
    else:  # either jsonl file or huggingface dataset
        train_goals, train_targets, test_goals, test_targets = get_goals_and_targets_jsonl(params)
        if params.word_blacklist_type in ["targets", "words_sim_with_targets"]:
            words = [word.lower() for sentence in train_targets for word in sentence.split(" ")]
            words += [word.capitalize() for sentence in train_targets for word in sentence.split(" ")]
            white_words = ["here", "of", "the", "is"]
            white_words += [w.capitalize() for w in white_words]
            words = [w for w in words if w not in white_words]
            params.word_blacklist = sorted(set(words))
        elif params.word_blacklist_type == "none":
            params.word_blacklist = []
        print(f"word blacklist type: {params.word_blacklist_type}")
        print(f"word blacklist: {params.word_blacklist}")

    workers, test_workers = get_workers(params)

    # NOTE: set the initialization of controls
    if params.instruction_init == "original" or params.instruction_init == "original_perturb":
        try:
            data = pd.read_json(params.train_data, lines=True)
        except:
            from datasets import load_dataset

            data = load_dataset(params.train_data)["train"].to_pandas()

        params.control_init = data[params.target_type].tolist()[params.data_offset]
        if params.instruction_init == "original_perturb":
            words = params.control_init.split()
            # words = random.shuffle(words)
            random.shuffle(words)
            params.control_init = " ".join(words)
        toks = workers[0].tokenizer.encode(params.control_init, add_special_tokens=False)
        if len(toks) < params.control_length:
            params.control_init = inject_exclamations(params.control_init, params.control_length)
        else:
            words = params.control_init.split()[: params.control_length]
            params.control_init = " ".join(words)

    elif params.instruction_init == "independent_sample":  # independently sampling
        data = pd.read_json(params.train_data, lines=True)
        instruction_list = data["instruction"].tolist()
        params.control_init = instruction_list[randint(0, len(instruction_list) - 1)]
    elif params.instruction_init == "placeholder":
        params.control_init = " ".join(["!"] * params.control_length)

    # NOTE: set the teacher_control for kd_gcg_attack
    teacher_control = None
    if params.attack == "kd_gcg":
        # NOTE: set the teacher_control for kd_gcg_attack
        data = pd.read_json(params.train_data, lines=True)
        teacher_control = data[params.control_type].tolist()[params.data_offset]
        assert teacher_control != params.control_init
        # set control_init the token length of teacher_control
        # params.control_init = " ".join(
        #     ["!"] * len(workers[0].tokenizer.encode(teacher_control, add_special_tokens=False))
        # )
        params.control_init = " ".join(["!"] * params.control_length)
        print(f"teacher_control: {teacher_control}")

    print(f"token length: {len(workers[0].tokenizer.encode(params.control_init, add_special_tokens=False))}")
    print(f"control init: {params.control_init}")

    small_workers = []
    if params.attack == "pbs_gcg":
        from configs.template import get_config as default_config

        _CONFIG_small = default_config()
        params_small = _CONFIG_small
        params_small.model_kwargs = [{"low_cpu_mem_usage": True}]
        params_small.conversation_templates = ["gpt2"]
        params_small.model_paths = ["openai-community/gpt2"]
        params_small.tokenizer_paths = ["openai-community/gpt2"]
        params_small.result_prefix = "results/gpt2"
        params_small.devices = ["cuda:0"]
        small_workers, _ = get_workers(params_small)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            teacher_control=teacher_control,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            mpa_word_blacklist=params.word_blacklist,
            mpa_word_blacklist_type=params.word_blacklist_type,
            mpa_word_blacklist_topk=params.word_blacklist_topk,
            mpa_goal_pos=params.goal_pos,
        )
    else:
        # train_goal + control -> train_target
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            small_workers=small_workers,
            control_init=params.control_init,
            teacher_control=teacher_control,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, "test_goals", []),
            test_targets=getattr(params, "test_targets", []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            mpa_word_blacklist=params.word_blacklist,
            mpa_word_blacklist_type=params.word_blacklist_type,
            mpa_goal_pos=params.goal_pos,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size,
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, "test_steps", 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
        early_stop_patience=params.early_stop_patience,
        probe_set=params.probe_set,
        filtered_set=params.filtered_set,
    )

    for worker in workers + test_workers + small_workers:
        worker.stop()


if __name__ == "__main__":
    time_start = time.time()
    app.run(main)
    print(f"Time elapsed: {time.time() - time_start:.2f}")
