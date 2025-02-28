import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.transfer = True
    config.logfile = ""
    config.num_train_models = 2

    config.progressive_goals = True
    config.stop_on_success = True
    config.tokenizer_paths = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "lmsys/vicuna-7b-v1.5",
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "lmsys/vicuna-7b-v1.5",
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True},
        {"low_cpu_mem_usage": True},
    ]
    config.conversation_templates = ["mistral", "vicuna"]
    config.devices = ["cuda:0", "cuda:1"]
    config.max_train_words = 32
    return config
