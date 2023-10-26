import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare
from attack.utils import Dict

import yaml
import datasets
from datasets import Image, Dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer

import os
os.environ['HTTP_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'
os.environ['HTTPS_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'

# Load config file
accelerator = Accelerator()

with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
    cfg = Dict(cfg)
cfg["cache_path"] = "./cache"

print(accelerator.device)

config = AutoConfig.from_pretrained(cfg.model_name)
config.use_cache = False
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(cfg.target_model, quantization_config=bnb_config,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=True,
                                                    config=config,
                                                    cache_dir=cfg["cache_path"])
model_type = config.to_dict()["model_type"]
if model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(cfg.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Load datasets
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
prompt_dataset = Dataset.from_dict(train_dataset[10000:20000])
prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)

model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

generated_dataset = {"text": []}

for text in tqdm(prompt_dataloader):
    prompt = (text["text"])
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(accelerator.device)
    clipped_ids = input_ids[:, :16]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    print(model(gen_tokens, labels=gen_tokens).loss)
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)

generated_dataset = Dataset.from_dict(generated_dataset)
save_dir = f"{cfg.cache_path}/{cfg.dataset_name}/{cfg.dataset_config_name}/refer@{cfg.model_name}/"
generated_dataset.save_to_disk(save_dir + f"{accelerator.device}")

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    concatenated_dataset = None
    for sub_dir in os.listdir(save_dir):
        data_path = os.path.join(save_dir, sub_dir)
        if os.path.isdir(data_path):
            if concatenated_dataset is None:
                concatenated_dataset = load_from_disk(data_path)
            else:
                dataset = load_from_disk(data_path)
                concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
    concatenated_dataset.save_to_disk(save_dir)