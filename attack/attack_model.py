import os
from accelerate.logging import get_logger
from attack import utils
from attack.utils import Dict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader

logger = get_logger(__name__, "INFO")

PATH = os.getcwd()

class AttackModel:
    def __init__(self, target_model, datasets, reference_model, shadow_model, cfg):
        self.target_model = target_model
        self.datasets = datasets
        self.kind = cfg['attack_kind']
        if shadow_model is not None and cfg['attack_kind'] == "nn":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model

    def llm_eval(self, model, input, cfg):
        outputs = []
        data_loader = DataLoader(
            dataset=input,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=32,
            pin_memory=torch.cuda.is_available()
        )
        pipeline = model
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        loss_function = getattr(self, cfg["loss_kind"]+"_loss")
        diffusion_steps = noise_scheduler.config.num_train_timesteps
        interval = diffusion_steps // cfg["diffusion_sample_number"]
        sample_steps = cfg["diffusion_sample_steps"]

        for iteration, batch in enumerate(data_loader):
            clean_images = batch["input"].cuda()
            batch_loss = np.zeros((cfg["eval_batch_size"], cfg["diffusion_sample_number"]))
            # start_time = time.time()
            for i, timestep in enumerate(sample_steps):
                if cfg["loss_kind"] == "ddpm":
                    loss = self.ddpm_loss(pipeline, clean_images, timestep)
                elif cfg["loss_kind"] == "ddim":
                    loss = self.ddim_loss(pipeline, clean_images, t_sec=timestep)
                batch_loss[:, i] = loss
            # print(f"time duration: {time.time() - start_time}s")
            outputs.append(batch_loss)
        output = np.concatenate(outputs, axis=0)
        return output

    def eval_perturb(self, model, dataset, cfg):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        per_losses = []
        ref_per_losses = []
        dataset_perturbation_function = self.image_dataset_perturbation if cfg["target_model"] == "vae" else self.norm_image_dataset_perturbation
        ori_dataset = deepcopy(dataset)
        ori_losses = self.llm_eval(model, ori_dataset, cfg)
        ref_ori_losses = self.llm_eval(self.reference_model, ori_dataset, cfg) if cfg["calibration"] else None
        strength = np.linspace(cfg['start_strength'], cfg['end_strength'], cfg['perturbation_number'])
        for i in tqdm(range(cfg["perturbation_number"])):
            per_dataset = dataset_perturbation_function(dataset, strength=strength[i])
            per_loss = self.llm_eval(model, per_dataset, cfg)
            per_losses.append(np.expand_dims(per_loss, -1))
            ref_per_loss = self.llm_eval(self.reference_model, per_dataset, cfg) if cfg["calibration"] else None
            try:
                ref_per_losses.append(np.expand_dims(ref_per_loss, -1))
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