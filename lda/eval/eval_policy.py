# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Literal
import os
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf

from LDA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from LDA.dataloader.lerobot_datasets import get_vla_dataset
from LDA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING
from LDA.model.framework.QwenGR00T import Qwen_GR00T
from LDA.model.framework.base_framework import baseframework
# from LDA.utils.eval_relative_eef import calc_mse_for_single_trajectory as calc_mse_for_single_trajectory_relative_eef
# from LDA.utils.eval_wo_postprocess import calc_mse_for_single_trajectory as calc_mse_for_single_trajectory_relative_eef
from LDA.model.framework import build_framework

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""

def main(config):
    np.random.seed(config.seed)
    data_cfg = config.datasets.vla_data
    model_cfg = cfg.framework.action_model
    dataset = get_vla_dataset(data_cfg, model_cfg=model_cfg, model_id=config.framework.qwenvl.base_vlm)  
    # policy: baseframework = build_framework(config)
    policy = baseframework.from_pretrained(pretrained_checkpoint = config.evaluation.model_path)
    policy.eval()
    policy.to("cuda")
    if config.is_delta_action:
        print("Delta Action Evaluation")
        from LDA.utils.eval_relative_eef import calc_mse_for_single_trajectory as calc_mse_for_single_trajectory_relative_eef
    else:
        from LDA.utils.eval_wo_postprocess import calc_mse_for_single_trajectory as calc_mse_for_single_trajectory_relative_eef

    all_tags = defaultdict(list)
    random_dataset_index = []
    for idx, ds in enumerate(dataset.datasets):
        tag = ds._metadata.embodiment_tag.value
        all_tags[tag].append(idx)
    for tag, indices in all_tags.items():
        if len(indices) < config.evaluation.trajs:
            random_dataset_index.extend(indices)
        else:
            random_dataset_index.extend(random.sample(indices, config.evaluation.trajs))                  
    print(f"Dataset Embodiment Tags: {all_tags}")

    print("Total trajectories:", len(dataset.datasets[0].trajectory_lengths))
    print("All trajectories:", dataset.datasets[0].trajectory_lengths)
    all_position_l1_loss = []
    all_orientation_l1_loss = []
    all_gripper_l1_loss = []
    all_hand_l1_loss = []
    
    # random sample evaluation.trajs 
    # for dataset_index in range(len(dataset.datasets)):
    for dataset_index in random_dataset_index:
        max_traj_id = len(dataset.datasets[dataset_index].trajectory_ids)
        traj_ids = np.random.choice(range(config.evaluation.start_traj, max_traj_id), size=1)
        # traj_ids = [agibot_subset_id[dataset_index]]
        dataset_name = dataset.datasets[dataset_index]._metadata.embodiment_tag.value
        plot_path = f"{config.evaluation.save_plot_path}/{dataset_name}"
        create_trajectory_video = config.evaluation.create_trajectory_video
        os.makedirs(plot_path, exist_ok=True)
        for traj_id in traj_ids:
            print("Running trajectory:", traj_id)
            traj_chunk = traj_id // 1000
            original_video_path = f"{dataset.datasets[dataset_index].dataset_path}/videos/chunk-{traj_chunk:03d}/observation.images.top_head/episode_{traj_id:06d}.mp4"
            video_output_path = f"{config.evaluation.video_output_path}/chunk-{traj_chunk:03d}_episode_{traj_id:06d}.mp4"
            eval_steps = dataset.datasets[dataset_index].trajectory_lengths[traj_id]
            l1_loss_dict = calc_mse_for_single_trajectory_relative_eef(
                policy,
                dataset.datasets[dataset_index],
                traj_id,
                steps=eval_steps,
                action_horizon=config.evaluation.action_horizon,
                plot=config.evaluation.plot,
                plot_state=config.evaluation.plot_state,
                save_plot_path=plot_path,
                create_trajectory_video=create_trajectory_video,
                video_output_path=video_output_path,
                original_video_path=original_video_path,
            )
            print("Position L1 lOSS:", l1_loss_dict["position_l1_loss"])
            print("Orientation L1 Loss:", l1_loss_dict["orientation_l1_loss"])
            if l1_loss_dict['gripper_l1_loss'] is not None:
                print("Gripper L1 Loss:", l1_loss_dict['gripper_l1_loss'])
            if l1_loss_dict['hand_l1_loss'] is not None:
                print("Hand L1 Loss:", l1_loss_dict['hand_l1_loss'])
            all_position_l1_loss.append(l1_loss_dict["position_l1_loss"])
            all_orientation_l1_loss.append(l1_loss_dict["orientation_l1_loss"])
            if l1_loss_dict["gripper_l1_loss"] is not None:
                all_gripper_l1_loss.append(l1_loss_dict["gripper_l1_loss"])
            if l1_loss_dict["hand_l1_loss"] is not None:
                all_hand_l1_loss.append(l1_loss_dict["hand_l1_loss"])
    for dataset_index in range(len(dataset.datasets)):
        print("Average Position L1 Loss across all trajs for dataset", dataset_index, ":", np.mean(all_position_l1_loss[dataset_index*config.evaluation.trajs:(dataset_index+1)*config.evaluation.trajs]))
        print("Average Orientation L1 Loss across all trajs for dataset", dataset_index, ":", np.mean(all_orientation_l1_loss[dataset_index*config.evaluation.trajs:(dataset_index+1)*config.evaluation.trajs]))
        if len(all_gripper_l1_loss) > 0:
            print("Average Gripper L1 Loss across all trajs for dataset", dataset_index, ":", np.mean(all_gripper_l1_loss[dataset_index*config.evaluation.trajs:(dataset_index+1)*config.evaluation.trajs]))
        if len(all_hand_l1_loss) > 0:
            print("Average Hand L1 Loss across all trajs for dataset", dataset_index, ":", np.mean(all_hand_l1_loss[dataset_index*config.evaluation.trajs:(dataset_index+1)*config.evaluation.trajs]))
    print("Average Position L1 Loss across all trajs:", np.mean(all_position_l1_loss))
    print("Average Orientation L1 Loss across all trajs:", np.mean(all_orientation_l1_loss))
    if len(all_gripper_l1_loss) > 0:
        print("Average Gripper L1 Loss across all trajs:", np.mean(all_gripper_l1_loss))
    if len(all_hand_l1_loss) > 0:
        print("Average Hand L1 Loss across all trajs:", np.mean(all_hand_l1_loss))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="LDA/config/training/LDA_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI config.evaluation to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    main(cfg)
