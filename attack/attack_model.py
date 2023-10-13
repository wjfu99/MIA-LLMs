import os
import random

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
import re
import seaborn as sns
from functools import partial
import spacy

logger = get_logger(__name__, "INFO")

PATH = os.getcwd()

accelerator = Accelerator()
class AttackModel:
    def __init__(self, target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=None, mask_tokenizer=None):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.kind = cfg['attack_kind']
        self.cfg = cfg
        if mask_model is not None:
            self.mask_model = mask_model
            self.mask_tokenizer = mask_tokenizer
            self.pattern = re.compile(r"<extra_id_\d+>")
        if shadow_model is not None and cfg['attack_kind'] == "nn":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model

    def get_raw_loss(self, outputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.detach().cpu().to(torch.float32).numpy()
        return loss
    def llm_eval(self, model, data_loader, cfg, idx_rate, perturb_fn=None, refer_model=None):
        model.eval()
        losses = []
        ref_losses = []
        token_lens = []
        for iteration, texts in enumerate(data_loader):
            texts = texts["text"]
            if cfg["maximum_samples"] is not None:
                if iteration * accelerator.num_processes >= cfg["maximum_samples"]:
                    break
            if perturb_fn is not None:
                texts = perturb_fn(texts)
            token_ids = self.tokenizer(texts, return_tensors="pt", padding=True).to(accelerator.device)
            labels = token_ids.input_ids
            outputs = model(**token_ids, labels=labels)
            ref_outputs = refer_model(**token_ids, labels=labels)
            loss = outputs.loss
            ref_loss = ref_outputs.loss
            token_lens.append(accelerator.gather(torch.tensor(token_ids.input_ids.size()[-1]).reshape(-1, 1).to(accelerator.device)).detach().cpu().numpy()) # TODO: may cause bug when running attacks in paralell.
            losses.append(accelerator.gather(loss.reshape(-1, 1)).detach().cpu().to(torch.float32).numpy())
            ref_losses.append(accelerator.gather(ref_loss.reshape(-1, 1)).detach().cpu().to(torch.float32).numpy())
            # print(f"{accelerator.device}@{texts}")
            # print(f"time duration: {time.time() - start_time}s")
        losses = np.concatenate(losses, axis=0)
        ref_losses = np.concatenate(ref_losses, axis=0)
        token_lens = np.concatenate(token_lens, axis=0)
        # token_lens = np.array(token_lens, dtype=np.int32)
        return losses, ref_losses, token_lens

    def eval_perturb(self, model, dataset, cfg):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        per_losses = []
        ref_per_losses = []
        per_token_lens = []
        ori_losses = []
        ref_ori_losses = []
        ori_token_lens = []
        ori_dataset = deepcopy(dataset)
        # TODO: the argument calibration is not work.
        strength = np.linspace(cfg['start_strength'], cfg['end_strength'], cfg['perturbation_number'])
        for i in tqdm(range(0, cfg["perturbation_number"])):
            idx_rate = i / cfg["perturbation_number"] * 0.7
            ori_loss, ref_ori_loss, ori_token_len = self.llm_eval(model, ori_dataset, cfg, idx_rate, refer_model=self.reference_model)
            ori_losses.append(ori_loss)
            ref_ori_losses.append(ref_ori_loss)
            ori_token_lens.append(ori_token_len)
            perturb_fn = partial(self.sentence_perturbation, idx_rate=idx_rate)
            sampled_per_losses = []
            sampled_ref_per_losses = []
            sampled_per_token_lens = []
            for _ in range(cfg["sample_number"]):
                per_loss, ref_per_loss, per_token_len = self.llm_eval(model, ori_dataset, cfg, idx_rate, perturb_fn=perturb_fn, refer_model=self.reference_model)
                sampled_per_losses.append(per_loss)
                sampled_ref_per_losses.append(ref_per_loss)
                sampled_per_token_lens.append(per_token_len)
            sampled_per_losses = np.concatenate(sampled_per_losses, axis=-1)
            sampled_ref_per_losses = np.concatenate(sampled_ref_per_losses, axis=-1)
            sampled_per_token_lens = np.concatenate(sampled_per_token_lens, axis=-1)
            per_losses.append(np.expand_dims(sampled_per_losses, axis=-1))
            ref_per_losses.append(np.expand_dims(sampled_ref_per_losses, axis=-1))
            per_token_lens.append(np.expand_dims(sampled_per_token_lens, axis=-1))
        ori_losses = np.concatenate(ori_losses, axis=-1)
        ref_ori_losses = np.concatenate(ref_ori_losses, axis=-1)
        ori_token_lens = np.concatenate(ori_token_lens, axis=-1)
        unnorm_ori_losses = (ori_token_lens - 1) * ori_losses
        unnorm_ref_ori_losses = (ori_token_lens - 1) * ref_ori_losses
        per_token_lens = np.concatenate(per_token_lens, axis=-1)
        per_losses = np.concatenate(per_losses, axis=-1)
        unnorm_per_losses = (per_token_lens - 1) * per_losses
        var_losses = per_losses - np.expand_dims(ori_losses, axis=-2)
        unnorm_var_losses = unnorm_per_losses - np.expand_dims(unnorm_ori_losses, axis=-2)
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg["calibration"] else None
        unnorm_ref_per_losses = (per_token_lens - 1) * ref_per_losses
        ref_var_losses = ref_per_losses - np.expand_dims(ref_ori_losses, axis=-2) if cfg["calibration"] else None
        unnorm_ref_var_losses = unnorm_ref_per_losses - np.expand_dims(unnorm_ref_ori_losses, axis=-2)


        output = (Dict(
            per_losses=per_losses,
            ori_losses=ori_losses,
            var_losses=var_losses,
            unnorm_ori_losses=unnorm_ori_losses,
            unnorm_per_losses=unnorm_per_losses,
            unnorm_var_losses=unnorm_var_losses
        ),
        Dict(
            ref_per_losses=ref_per_losses,
            ref_ori_losses=ref_ori_losses,
            ref_var_losses=ref_var_losses,
            unnorm_ref_ori_losses=unnorm_ref_ori_losses,
            unnorm_ref_per_losses=unnorm_ref_per_losses,
            unnorm_ref_var_losses=unnorm_ref_var_losses
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
            if accelerator.is_main_process:
                utils.save_dict_to_npz(mem_feat, mem_path)
                if cfg["calibration"]:
                    utils.save_dict_to_npz(ref_mem_feat, ref_mem_path)

            logger.info("Generating feature vectors for non-member data...")
            nonmem_feat, ref_nonmem_feat = self.eval_perturb(target_model, nonmem_data, cfg)
            if accelerator.is_main_process:
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
            mem_feat = info_dict.mem_feat.var_losses / np.expand_dims(info_dict.mem_feat.ori_losses, -2)\
                       - info_dict.ref_mem_feat.ref_var_losses / np.expand_dims(info_dict.ref_mem_feat.ref_ori_losses, -2)
            nonmem_feat = info_dict.nonmem_feat.var_losses / np.expand_dims(info_dict.nonmem_feat.ori_losses, -2)\
                       - info_dict.ref_nonmem_feat.ref_var_losses / np.expand_dims(info_dict.ref_nonmem_feat.ref_ori_losses, -2)
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
            mem_feat = mem_feat[:, :, 7]
            nonmem_feat = nonmem_feat[:, :, 7]
            mem_feat[np.isnan(mem_feat)] = 0
            nonmem_feat[np.isnan(nonmem_feat)] = 0
            # feat = np.concatenate([info_dict.mem_feat.ori_losses - info_dict.ref_mem_feat.ref_ori_losses, info_dict.nonmem_feat.ori_losses - info_dict.ref_nonmem_feat.ref_ori_losses])
            # feat = np.concatenate([info_dict.mem_feat.per_losses.mean(axis=(-1)), info_dict.nonmem_feat.per_losses.mean(axis=(-1))])
            feat = np.concatenate([mem_feat.mean(axis=(-1)), nonmem_feat.mean(axis=(-1))])
            # feat = np.concatenate([info_dict.mem_feat.var_losses.min(axis=-1), info_dict.nonmem_feat.var_losses.min(axis=-1)])
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

    # @staticmethod
    # def sentence_perturbation(batch, tokenizer):
    #
    #     ### Demo
    #     ids = batch["input_ids"][0]
    #     text_list = []
    #     ids_list = []
    #     for _ in range(100):
    #         text = tokenizer.decode(ids)
    #         text_list.append(text)
    #         ids = tokenizer(text, truncation=True)["input_ids"]
    #         ids_list.append(ids)
    #     # aug = naw.RandomWordAug(action="swap", aug_p=0.2)
    #     # aug = naw.SynonymAug(aug_src="wordnet", aug_p=0)
    #     # sentence = tokenizer.decode(batch["input_ids"][0])
    #     # # perturb_sentence = aug.augment(sentence)
    #     # perturb_sentence = sentence
    #     # perturb_ids = tokenizer(perturb_sentence, truncation=True)["input_ids"]
    #     perturb_ids = deepcopy(batch["input_ids"])
    #     for i in range(batch["input_ids"].shape[1]):
    #         if batch["input_ids"][0, i].item() != tokenizer.eos_token_id and random.random() < 0.01:
    #             perturb_ids[0, i] = random.randint(0, 50255)
    #
    #     return {
    #                 "input_ids": perturb_ids,
    #                 "labels": perturb_ids
    #             }

    def tokenize_and_mask(self, text, span_length, pct, idx_rate, ceil_pct=False):
        cfg = self.cfg
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'
        perturb_start_idx = int(len(tokens) * idx_rate)

        n_spans = pct * len(tokens) / (span_length + cfg.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(perturb_start_idx, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - cfg.buffer_size)
            search_end = min(len(tokens), end + cfg.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def mask_noun(self, text, pct, ceil_pct=False):
        cfg = self.cfg
        # tokens = text.split(' ')
        sp = spacy.load('en_core_web_sm')
        doc = sp(text)
        mask_string = '<<<mask>>>'

        n_spans = pct * len(doc) / (cfg.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        tokens = []
        for idx, token in enumerate(doc):
            if len(tokens) > 0:
                if token.pos_ in ['NOUN', "PROPN"] and f'<extra_id_{num_filled-1}>' not in tokens[-cfg.buffer_size:]:
                    tokens.append(f'<extra_id_{num_filled}>')
                    num_filled += 1
                else:
                    tokens.append(token.text)
            else:
                tokens.append(token.text)
        # assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    @staticmethod
    def count_masks(texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, texts):
        """
        predict mask tokens with the mask model.
        :param texts:
        :return:
        """
        cfg = self.cfg
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(accelerator.device)
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=cfg.mask_top_p,
                                      num_return_sequences=1, eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills_(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text = text[:text.index(f"<extra_id_{fill_idx}>")+1]
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
                    tokens[idx] = text

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def sentence_perturbation(self, texts, idx_rate):
        cfg = self.cfg
        if cfg.only_noun:
            masked_texts = [self.mask_noun(x, cfg.pct) for x in texts]
        else:
            masked_texts = [self.tokenize_and_mask(x, cfg.span_length, cfg.pct, idx_rate, cfg.ceil_pct) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            if cfg.only_noun:
                masked_texts = [self.mask_noun(x, cfg.pct) for idx, x in enumerate(texts) if idx in idxs]
            else:
                masked_texts = [self.tokenize_and_mask(x, cfg.span_length, cfg.pct, idx_rate, cfg.ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts

    def mask_idx(self, text, span_length, idx_rate):
        cfg = self.cfg
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        start = int((len(tokens) - span_length) * idx_rate)
        end = start + span_length
        search_start = max(0, start - cfg.buffer_size)
        search_end = min(len(tokens), end + cfg.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx_rate, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx_rate] = f'<extra_id_{num_filled}>'
                num_filled += 1
        text = ' '.join(tokens)
        return text

    def mask_tokens(self, text, span_length, idx_rate):
        mask_string = '<extra_id_0>'
        input_ids = self.tokenizer.encode(text)
        eos = int((len(input_ids)) * idx_rate)
        input_ids1 = input_ids[:eos-span_length]
        input_ids2 = input_ids[eos:]
        text1 = self.tokenizer.decode(input_ids1)
        text2 = self.tokenizer.decode(input_ids2)
        text = ' '.join([text1, mask_string, text2])

        return text

    def sentence_idx_perturbation(self, texts, idx_rate):
        cfg = self.cfg
        masked_texts = [self.mask_tokens(x, cfg.span_length, idx_rate) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills_(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.mask_tokens(x, cfg.span_length, idx_rate) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills_(masked_texts, extracted_fills)
            for idx_rate, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx_rate] = x
            attempts += 1
        return perturbed_texts

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

    @staticmethod
    def distinguishability_plot(mem, non_mem):
        sns.set_theme()
        mem_color = "indianred"
        non_mem_color = "forestgreen"
        sns.kdeplot(mem, fill=True, color=mem_color, alpha=0.5)
        sns.kdeplot(non_mem, fill=True, color=non_mem_color, alpha=0.5)

        mem_mean = round(mem.mean(), 2)
        mem_std = round(mem.std(), 2)
        non_mem_mean = round(non_mem.mean(), 2)
        non_mem_std = round(non_mem.std(), 2)

        # plt.xlabel(r"${\mathcal{F}}({x}, \theta)$", fontsize=22, labelpad=10)
        plt.xlabel(r"$\Delta \widehat{p}_{\theta}$", fontsize=22, labelpad=10)
        plt.ylabel('Density', fontsize=22, labelpad=10)
        plt.legend(['Member', 'Non-member'], fontsize=20, loc='upper right')
        # plt.xlim([-0.6, 0.9])
        mem_text = '\n'.join((
                    r'$\mu_{Mem}=%.2f$' % (mem_mean, ),
                    r'$\sigma_{Mem}=%.2f$' % (mem_std, )))
        non_mem_text = '\n'.join((
                    r'$\mu_{Non}=%.2f$' % (non_mem_mean, ),
                    r'$\sigma_{Non}=%.2f$' % (non_mem_std, )))
        mem_props = dict(boxstyle='round', facecolor=mem_color, alpha=0.15, edgecolor='black')
        non_mem_props = dict(boxstyle='round', facecolor=non_mem_color, alpha=0.15, edgecolor='black')

        plt.tick_params(labelsize=16)
        # plt.text(0.63, 0.25, mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=mem_props)
        # plt.text(0.04, 0.6, non_mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=non_mem_props)

        plt.tight_layout()
        # plt.savefig("distinguishability-diffusion-our.pdf", format="pdf", bbox_inches="tight")
        plt.show()