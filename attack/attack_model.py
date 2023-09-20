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
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from itertools import cycle
import matplotlib.pyplot as plt

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
            if iteration == cfg["maximum_samples"]:
                break
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
        var_losses = per_losses - ori_losses
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg["calibration"] else None
        ref_var_losses = ref_per_losses - ref_ori_losses if cfg["calibration"] else None

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
        data_path = os.path.join(PATH, cfg["attack_data_path"], f"attack_data_{cfg['model_name']}@{cfg['dataset_name']}")
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

    def feat_prepare(self, info_dict, cfg):
        # mem_info = info_dict.mem_feat
        # ref_mem_info = info_dict.ref_mem_feat
        if cfg["calibration"]:
            mem_feat = info_dict.mem_feat.var_losses / info_dict.mem_feat.ori_losses\
                       - info_dict.ref_mem_feat.ref_var_losses / info_dict.ref_mem_feat.ref_ori_losses
            nonmem_feat = info_dict.nonmem_feat.var_losses / info_dict.nonmem_feat.ori_losses\
                       - info_dict.ref_nonmem_feat.ref_var_losses / info_dict.ref_nonmem_feat.ref_ori_losses
            # gen_feat = info_dict.gen_feat.var_losses / info_dict.gen_feat.ori_losses[:, :, None] \
            #               - info_dict.ref_gen_feat.ref_var_losses / info_dict.ref_gen_feat.ref_ori_losses[:, :, None]
        else:
            mem_feat = info_dict.mem_feat.var_losses / info_dict.mem_feat.ori_losses
            nonmem_feat = info_dict.nonmem_feat.var_losses / info_dict.nonmem_feat.ori_losses
            # gen_feat = info_dict.gen_feat.var_losses / info_dict.gen_feat.ori_losses[:, :, None]
        # if cfg["target_model"] == "diffusion":
        #     mem_feat = mem_feat[:, 2, :]
        #     nonmem_feat = nonmem_feat[:, 2, :]
            # gen_feat = gen_feat[:, 2, :]

        if cfg["attack_kind"] == "stat":
            mem_feat[np.isnan(mem_feat)] = 0
            nonmem_feat[np.isnan(nonmem_feat)] = 0
            # feat = np.concatenate([info_dict.mem_feat.ori_losses, info_dict.nonmem_feat.ori_losses])
            # feat = np.concatenate([info_dict.mem_feat.per_losses.mean(axis=(-1)), info_dict.nonmem_feat.per_losses.mean(axis=(-1))])
            feat = np.concatenate([mem_feat.mean(axis=(-1)), nonmem_feat.mean(axis=(-1))])
            ground_truth = np.concatenate([np.zeros(mem_feat.shape[0]), np.ones(nonmem_feat.shape[0])]).astype(int)

        elif cfg["attack_kind"] == "nn":
            # mem_freq = self.frequency(mem_feat, split=100)
            # nonmem_freq = self.frequency(nonmem_feat, split=100)
            # mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_freq, nonmem_freq)
            mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_feat, nonmem_feat)
            if cfg["target_model"] == "vae":
                mem_feat.sort(axis=1)
                nonmem_feat.sort(axis=1)
            feat = torch.cat([mem_feat, nonmem_feat])
            feat[torch.isnan(feat)] = 0
            # if cfg["target_model"] == "diffusion":
            feat = feat.unsqueeze(1)
            ground_truth = torch.cat([torch.zeros(mem_feat.shape[0]), torch.ones(nonmem_feat.shape[0])]).type(torch.LongTensor).cuda()
        return feat, ground_truth

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
        # aug = naw.RandomWordAug(action="swap", aug_p=0.2)
        aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.3)
        sentence = tokenizer.decode(batch["input_ids"][0])
        perturb_sentence = aug.augment(sentence)
        perturb_ids = tokenizer(perturb_sentence, truncation=True)["input_ids"]
        return {
                    "input_ids": torch.LongTensor(perturb_ids).to(accelerator.device),
                    "labels": torch.LongTensor(perturb_ids).to(accelerator.device)
                }

    @staticmethod
    def eval_attack(y_true, y_scores, plot=True, path=None):
        if type(y_true) == torch.Tensor:
            y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        if path is not None:
            np.savez(path, fpr=fpr, tpr=tpr)
        auc_score = roc_auc_score(y_true, y_scores)
        logger.info(f"AUC on the target model: {auc_score}")

        # Finding the threshold point where FPR + TPR equals 1
        threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        logger.info(f"ASR on the target model: {threshold_point}")

        # Finding the threshold point where FPR + TPR equals 1
        tpr_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
        logger.info(f"TPR@1%FPR on the target model: {tpr_1fpr}")


        if plot:
            # plot the ROC curve
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score}; ASR = {threshold_point})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            # plot the no-skill line for reference
            plt.plot([0, 1], [0, 1], linestyle='--')
            # show the plot
            plt.show()