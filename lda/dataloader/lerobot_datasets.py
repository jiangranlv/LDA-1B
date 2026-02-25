# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].
import torch
import numpy as np
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from lda.dataloader.gr00t_lerobot.video_gen_datasets import VideoTaskSingleDataset
from lda.dataloader.gr00t_lerobot.mixtures import get_dataset_mixtures
from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from lda.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
TRAINING_TASKS = ["policy", "forward_dynamics", "inverse_dynamics", "video_gen"]
VIDEOGEN_DATASET = ["egocentric_10k", "taste_rob", "rh20t"]
def collate_fn(batch):
    return batch
def collate_fn_Qwen2_5(batch, processor):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        values = [elem[key] for elem in batch]
        if values[0] is None:
            collated_batch[key] = None
            continue
        if key == "vlm_inputs":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            vlm_inputs = processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in vlm_inputs.items():
                collated_batch[f'vlm_{k}'] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            collated_batch[key] = torch.cat(values)
        elif key in ("intrinsic", "extrinsic", "assigned_task", "lang"):
            collated_batch[key] = values
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            collated_batch[key] = torch.from_numpy(np.stack(values))
    return collated_batch
def collate_fn_Qwen3(batch, processor):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        values = [elem[key] for elem in batch]
        if values[0] is None:
            collated_batch[key] = None
            continue
        if key == "vlm_inputs":
            messages = []
            for v in values:
                messages.append(v)
            vlm_inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            padding=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                            )
            for k, v in vlm_inputs.items():
                collated_batch[f'vlm_{k}'] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            collated_batch[key] = torch.cat(values)
        elif key in ("intrinsic", "extrinsic", "assigned_task", "lang"):
            collated_batch[key] = values
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            collated_batch[key] = torch.from_numpy(np.stack(values))
    return collated_batch
def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    CoT_prompt: str = None, 
) -> LeRobotSingleDataset | VideoTaskSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type] # in data_config.py
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    if hasattr(data_config, "video_backend"):
        video_backend = data_config.video_backend
    else:
        video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    if hasattr(data_config, "history_action_indices"):
        history_action_indices = data_config.history_action_indices
    else:
        history_action_indices = None
    if hasattr(data_config, "img_interval"):
        img_interval = data_config.img_interval
    else:
        img_interval = 1
    if robot_type == "egocentric_10k":
        print("==============")
        print(f"Egocentric-10K is contained in dataset")
        return VideoTaskSingleDataset(
            trajectory_root=dataset_path,
            modality_configs=modality_config,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
            history_action_indices=history_action_indices,
            metadata_cache_path="/mnt/project/world_model/data/HumanData/Egocentric-10K/metadata.pkl"
        )
    else:
        return LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend, 
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            img_interval=img_interval,
            history_action_indices=history_action_indices,
            CoT_prompt=CoT_prompt,
        )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    model_id: str = 'Qwen2.5',
    **kwargs: dict,
) -> tuple[LeRobotMixtureDataset, LeRobotMixtureDataset] | LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    CoT_prompt = data_cfg.get("CoT_prompt", None)
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    model_cfg = kwargs.get("model_cfg", None)
    if model_cfg is not None:
        state_dim = model_cfg.get("state_dim", None)
        action_dim = model_cfg.get("action_dim", None)
    # mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    mixture_spec = get_dataset_mixtures(data_root_dir, data_mix)
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    all_dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        if d_name in VIDEOGEN_DATASET:
            all_dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg, CoT_prompt=CoT_prompt), d_weight))
        else:
            dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg, CoT_prompt=CoT_prompt), d_weight))
            all_dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg, CoT_prompt=CoT_prompt), d_weight))

    w_action_dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        use_delta_action=data_cfg.get("use_delta_action", False),
        data_cfg=data_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        model_id=model_id,
        **kwargs,
    )
    all_dataset = LeRobotMixtureDataset(
        all_dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        use_delta_action=data_cfg.get("use_delta_action", False),
        data_cfg=data_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        model_id=model_id,
        **kwargs,
    )
    
    if len(dataset_mixture) == len(all_dataset_mixture):
        return w_action_dataset
    else:
        return w_action_dataset, all_dataset



if __name__ == "__main__":

    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./lda/config/training/lda_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "./examples/MultiRobot/train_files/lda_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass