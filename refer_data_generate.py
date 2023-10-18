import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random

from data.prepare import dataset_prepare
from attack.utils import Dict

import yaml
import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig

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

config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
config.use_cache = False
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained("./ft_llms/target_model_gptj/checkpoint-200", quantization_config=bnb_config,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=True,
                                                    config=config)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# Load datasets
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
prompt_dataset = Dataset.from_dict(train_dataset[6000:12000])
prompt_dataloader = DataLoader(prompt_dataset, batch_size=10)

model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

generated_dataset = {"text": []}

for text in tqdm(prompt_dataloader):
    prompt = (text["text"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
    clipped_ids = input_ids[:, :16]
    gen_tokens = model.generate(
        clipped_ids,
        num_beams=2,
        do_sample=True,
        # temperature=0.3,
        max_length=128,
    )
    print(model(gen_tokens, labels=gen_tokens).loss)
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)

generated_dataset = Dataset.from_dict(generated_dataset)
generated_dataset.save_to_disk(f"./cache/wikitext/refer_dataset_gptj/{accelerator.device}")