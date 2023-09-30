import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random

from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict

import yaml
import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig

os.environ['HTTP_PROXY'] = 'http://10.22.80.36:7890'
os.environ['HTTPS_PROXY'] = 'http://10.22.80.36:7890'

# Load config file
with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
    cfg = Dict(cfg)

# Add Logger
accelerator = Accelerator()
logger = get_logger(__name__, "INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Automatically select the freest GPU.
# os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
# memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
# os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_available))
# device = "cuda" + ":" + str(np.argmax(memory_available))
# torch.cuda.set_device(device)

# Fix the random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## Load generation models.
if not cfg["load_attack_data"]:
    config = AutoConfig.from_pretrained(cfg["model_name"])
    config.use_cache = False
    bnb_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target_model = AutoModelForCausalLM.from_pretrained(cfg["target_model"], quantization_config=bnb_config,
                                                        torch_dtype=torch_dtype,
                                                        local_files_only=True,
                                                        config=config).to(accelerator.device)
    reference_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=bnb_config,
                                                           torch_dtype=torch_dtype,
                                                           config=config).to(accelerator.device)

    logger.info("Successfully load models")

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                              add_bos_token=cfg["add_bos_token"], use_fast=True)

    if cfg["pad_token_id"] is not None:
        logger.info("Using pad token id %d", cfg["pad_token_id"])
        tokenizer.pad_token_id = cfg["pad_token_id"]

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    logger.info("Successfully load datasets!")

    # Prepare dataloade
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"])
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg["eval_batch_size"])

    shadow_model = None
    int8_kwargs = {}
    half_kwargs = {}
    if cfg["int8"]:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif cfg["half"]:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["mask_filling_model_name"], **int8_kwargs, **half_kwargs).to(accelerator.device)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(cfg["mask_filling_model_name"], model_max_length=n_positions)

    # Prepare everything with accelerator
    train_dataloader, eval_dataloader = (
        accelerator.prepare(
            train_dataloader,
            eval_dataloader,
    ))
else:
    target_model = None
    reference_model = None
    shadow_model = None
    mask_model = None
    train_dataloader = None
    eval_dataloader = None
    tokenizer = None
    mask_tokenizer = None

"""
losses = []
losses_ref = []
for step, batch in tqdm(enumerate(eval_dataloader)):
    with torch.no_grad():
        outputs = target_model(**batch)
        outputs_ref = reference_model(**batch)

    loss = outputs.loss
    losses.append(accelerator.gather(loss.reshape(-1, 1)))
    loss_ref = outputs_ref.loss
    losses_ref.append(accelerator.gather(loss_ref.reshape(-1, 1)))
losses = torch.cat(losses, dim=0)
losses_ref = torch.cat(losses_ref, dim=0)
sorted_ratio = sorted([l-l_ref for l,l_ref in zip (losses,losses_ref)])
threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]


losses = []
losses_ref = []
for step, batch in enumerate(train_dataloader):
    with torch.no_grad():
        outputs = target_model(**batch)

    loss = outputs.loss

    losses.append(accelerator.gather(loss.reshape(-1, 1)))
    with torch.no_grad():
        outputs_ref = reference_model(**batch)
    loss_ref = outputs_ref.loss
    losses_ref.append(accelerator.gather(loss_ref.reshape(-1, 1)))

accelerator.wait_for_everyone()
losses = torch.cat(losses, dim=0)

losses_ref = torch.cat(losses_ref, dim=0)
lr_rat = [l - l_r for l, l_r in zip(losses, losses_ref)]
guess_cor_ref = sum([1 for sample in lr_rat if sample < threshold_ref])
print("correct cnt  ref is: ", guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
"""

datasets = {
    "target": {
        "train": train_dataloader,
        "valid": eval_dataloader
    },
    # "shadow": {
    #     "train": Dataset.from_dict(all_dataset[random.sample(range(35000, 65000), cfg["sample_number"])]),
    #     "valid": Dataset.from_dict(all_dataset[random.sample(range(65000, 70000), cfg["sample_number"])])
    # },
    # "reference": {
    #     "train": Dataset.from_dict(all_dataset[random.sample(range(70000, 100000), cfg["sample_number"])]),
    #     "valid": Dataset.from_dict(all_dataset[random.sample(range(100000, 105000), cfg["sample_number"])])
    # }
}


attack_model = AttackModel(target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
attack_model.conduct_attack(cfg=cfg)
