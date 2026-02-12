# Copyright 2025 LDA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from token import OP
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch.distributed as dist
import time


from deployment.model_server.tools.image_tools import to_pil_preserve
from LDA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from LDA.model.framework.base_framework import baseframework
from LDA.model.modules.vlm import get_vlm_model
from LDA.model.modules.action_model.UWM_ActionHeader import get_action_model, FlowmatchingActionHead
from LDA.training.trainer_utils.trainer_tools import resize_images
from LDA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("UWM")
class UWM(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, action_dim]
        actions_mask = [example["action_mask"] for example in examples]
        batch_future_images = [example["future_image"] for example in examples]  # [B，[PLT]]
        curr_images = np.array(batch_images).transpose(0, 1, 4, 2, 3)
        future_images = np.array(batch_future_images).transpose(0, 1, 4, 2, 3)
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        embodiment_ids = [example["embodiment_id"] for example in examples]
        
        # Step 4: Action Expert Forward and Loss
        # with torch.autocast("cuda", dtype=torch.float32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.from_numpy(np.array(actions)) # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            actions_mask = torch.from_numpy(np.array(actions_mask)).to(self.action_model.device)
            actions_target_mask = actions_mask[:, -(self.future_action_window_size+1):, :]
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1).to(self.action_model.device)
            actions_target_mask_repeated = actions_target_mask.repeat(repeated_diffusion_steps, 1, 1)

            embodiment_ids = torch.tensor(
                np.array(embodiment_ids), device=self.action_model.device, dtype=torch.int32
            )
            embodiment_ids_repeated = embodiment_ids.repeat(repeated_diffusion_steps)

            curr_images_repeated = torch.from_numpy(curr_images).repeat(repeated_diffusion_steps, 1, 1, 1, 1).to(self.action_model.device)
            future_images_repeated = torch.from_numpy(future_images).repeat(repeated_diffusion_steps, 1, 1, 1, 1).to(self.action_model.device)
            state_repeated = None
            if state is not None:
                state = torch.from_numpy(np.array(state))
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1).to(self.action_model.device)
            instructions = instructions * repeated_diffusion_steps
            output_dict = self.action_model(
                actions=actions_target_repeated, 
                action_mask=actions_target_mask_repeated,
                state=state_repeated, 
                future_imgs=future_images_repeated, 
                curr_imgs=curr_images_repeated, 
                embodiment_id=embodiment_ids_repeated, 
                langs=instructions)  # (B, chunk_len, action_dim)

        return output_dict

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        curr_imgs = torch.from_numpy(np.array([example["image"] for example in examples]))
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        embodiment_ids = [example["embodiment_id"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)


        state = torch.from_numpy(np.array(state)).to(self.action_model.device) if state is not None else None
        # state = examples['state'].to(self.action_model.device, ) if 'state' in examples else None
        curr_imgs = curr_imgs.to(self.action_model.device)
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(state, curr_imgs, 
                                                            embodiment_ids, instructions)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().float().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./LDA/config/training/LDA_cotrain_agibot.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "/mnt/project/world_model/pretrained/vlm/Florence-2-large"
     
    model: Qwen_MMDiT = Qwen_MMDiT(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    future_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "future_image": [future_image, future_image], # two views
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from LDA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
