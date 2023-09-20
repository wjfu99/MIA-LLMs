import os
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from attack import utils
from attack.utils import Dict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

logger = get_logger(__name__, "INFO")

PATH = os.getcwd()

accelerator = Accelerator()
class AttackModel:
    def __init__(self, target_model, tokenizer, datasets, reference_model, shadow_model, cfg):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.kind = cfg['attack_kind']
        if shadow_model is not None and cfg['attack_kind'] == "nn":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model

    def llm_eval(self, model, data_loader, cfg, perturb_fn=None):
        model.eval()
        sample_steps = cfg["extensive_per_num"]
        losses = []
        for iteration, batch in enumerate(data_loader):
            if perturb_fn is not None:
                batch = perturb_fn(batch, self.tokenizer)
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.reshape(-1, 1)).detach().cpu().to(torch.float32).numpy())
            # print(f"time duration: {time.time() - start_time}s")
        losses = np.concatenate(losses, axis=0)
        return losses

    def eval_perturb(self, model, dataset, cfg):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        per_losses = []
        ref_per_losses = []
        ori_dataset = deepcopy(dataset)
        ori_losses = self.llm_eval(model, ori_dataset, cfg)
        ref_ori_losses = self.llm_eval(self.reference_model, ori_dataset, cfg) if cfg["calibration"] else None
        strength = np.linspace(cfg['start_strength'], cfg['end_strength'], cfg['perturbation_number'])
        for i in tqdm(range(cfg["perturbation_number"])):
            per_loss = self.llm_eval(model, ori_dataset, cfg, perturb_fn=self.sentence_perturbation)
            per_losses.append(per_loss)
            ref_per_loss = self.llm_eval(self.reference_model, ori_dataset, cfg, perturb_fn=self.sentence_perturbation) if cfg["calibration"] else None
            try:
                ref_per_losses.append(ref_per_loss)
            except:
                pass
        per_losses = np.concatenate(per_losses, axis=-1)
        var_losses = per_losses - np.expand_dims(ori_losses, -1)
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg["calibration"] else None
        ref_var_losses = ref_per_losses - np.expand_dims(ref_ori_losses, -1) if cfg["calibration"] else None

        output = (Dict(
            per_losses=per_losses,
            ori_losses=ori_losses,
            var_losses=var_losses,
        ),
        Dict(
            ref_per_losses=ref_per_losses,
            ref_ori_losses=ref_ori_losses,
            ref_var_losses=ref_var_losses,
        ))
        return output

    def data_prepare(self, kind, cfg):
        logger.info("Preparing data...")
        data_path = os.path.join(PATH, cfg["attack_data_path"], f"attack_data_{cfg['target_model']}@{cfg['dataset_name']}")
        target_model = getattr(self, kind + "_model")
        mem_data = self.datasets[kind]["train"]
        nonmem_data = self.datasets[kind]["valid"]

        mem_path = os.path.join(data_path, kind, "mem_feat.npz")
        nonmem_path = os.path.join(data_path, kind, "nonmen_feat.npz")
        ref_mem_path = os.path.join(data_path, kind, "ref_mem_feat.npz")
        ref_nonmem_path = os.path.join(data_path, kind, "ref_nonmen_feat.npz")

        pathlist = (mem_path, nonmem_path, ref_mem_path, ref_nonmem_path) if cfg["calibration"] else (mem_path, nonmem_path)

        if not utils.check_files_exist(*pathlist) or not cfg["load_attack_data"]:

            logger.info("Generating feature vectors for member data...")
            mem_feat, ref_mem_feat = self.eval_perturb(target_model, mem_data, cfg)
            utils.save_dict_to_npz(mem_feat, mem_path)
            if cfg["calibration"]:
                utils.save_dict_to_npz(ref_mem_feat, ref_mem_path)

            logger.info("Generating feature vectors for non-member data...")
            nonmem_feat, ref_nonmem_feat = self.eval_perturb(target_model, nonmem_data, cfg)
            utils.save_dict_to_npz(nonmem_feat, nonmem_path)
            if cfg["calibration"]:
                utils.save_dict_to_npz(ref_nonmem_feat, ref_nonmem_path)

            logger.info("Saving feature vectors...")

        else:
            logger.info("Loading feature vectors...")
            mem_feat = utils.load_dict_from_npz(mem_path)
            ref_mem_feat = utils.load_dict_from_npz(ref_mem_path) if cfg["calibration"] else None
            nonmem_feat = utils.load_dict_from_npz(nonmem_path)
            ref_nonmem_feat = utils.load_dict_from_npz(ref_nonmem_path) if cfg["calibration"] else None

        logger.info("Data preparation complete.")

        return Dict(
            mem_feat=mem_feat,
            nonmem_feat=nonmem_feat,
            ref_mem_feat=ref_mem_feat,
            ref_nonmem_feat=ref_nonmem_feat,
                    )

    def conduct_attack(self, cfg):
        save_path = os.path.join(PATH, cfg["attack_data_path"], f"attack_data_{cfg['model_name']}@{cfg['dataset_name']}",
                                 f"roc_{cfg['attack_kind']}.npz")
        if cfg["attack_kind"] == 'nn':
            if not self.is_model_training:
                self.attack_model_training(cfg)
            attack_model = self.attack_model
            raw_info = self.data_prepare("target", cfg)
            feat, ground_truth = self.feat_prepare(raw_info, cfg)
            predict = attack_model(feat)
            # predict, ground_truth = utils.tensor_to_ndarray(predict, ground_truth)
            self.eval_attack(ground_truth, predict[:, 1], path=save_path)
        elif cfg["attack_kind"] == 'stat':
            raw_info = self.data_prepare("target", cfg)
            feat, ground_truth = self.feat_prepare(raw_info, cfg)
            # self.distinguishability_plot(raw_info['mem_feat']['ori_losses'].mean(-1),
            #                              raw_info['nonmem_feat']['ori_losses'].mean(-1))
            # self.distinguishability_plot(feat[:1000], feat[-1000:])
            self.eval_attack(ground_truth, -feat, path=save_path)

    @staticmethod
    def sentence_perturbation(batch, tokenizer):
        aug = naw.RandomWordAug(action="swap", aug_p=0.2)
        sentence = tokenizer.decode(batch["input_ids"][0])
        perturb_sentence = aug.augment(sentence)
        perturb_ids = tokenizer(perturb_sentence, truncation=True)["input_ids"]
        return {
                    "input_ids": torch.LongTensor(perturb_ids).to(accelerator.device),
                    "labels": torch.LongTensor(perturb_ids).to(accelerator.device)
                }