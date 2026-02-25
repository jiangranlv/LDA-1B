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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio

from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from lda.model.framework.base_framework import baseframework
import h5py
import torch
from pytorch3d.transforms import euler_angles_to_matrix
# numpy print precision settings 3, dont use exponential notation

from lda.utils.rotation_convert import delta2abs, calculate_delta_eef
from lda.dataloader.lerobot_datasets import collate_fn
from lda.utils.eval_relative_eef import create_action_trajectory_video

AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
PIXEL_EPS = 1e-9
PRED_COLOR_BASE = {"leftHand": (0, 0, 255), "rightHand": (0, 0, 255)} # 
GT_COLOR_BASE = {"leftHand": (255, 255, 0), "rightHand": (255, 255, 0)} # 
AXIS_LENGTH = 0.1
AXIS_THICKNESS = 2
POINT_RADIUS = 5

WO_GRIPPER_DATASET = ['human', 'egovla']
ROBOCASA_DATASET = ["robocasa", "gr1"]

np.set_printoptions(precision=3, suppress=True)

def get_abs_eef(action_chunk, data_point, wo_gripper):

    left_action_delta_pose = np.concatenate([action_chunk["action.left_eef_position"], action_chunk["action.left_eef_rotation"]], axis=1)
    left_initial_action_pose = np.concatenate([data_point["action"][0, :3], data_point["action"][0, 3:6]])
    left_action_abs_pose = delta2abs(left_action_delta_pose, left_initial_action_pose)
    right_action_delta_pose = np.concatenate([action_chunk["action.right_eef_position"], action_chunk["action.right_eef_rotation"]], axis=1)
    if wo_gripper:
        right_initial_action_pose = np.concatenate([data_point["action"][0, 6:9], data_point["action"][0, 9:12]])
    else:
        right_initial_action_pose = np.concatenate([data_point["action"][0, 7:10], data_point["action"][0, 10:13]])
    right_action_abs_pose = delta2abs(right_action_delta_pose, right_initial_action_pose)

    return left_action_abs_pose, right_action_abs_pose

def angular_diff(a, b):
    """
    Compute the smallest signed difference between two angles (in radians).
    Result is in [-pi, pi).
    
    Args:
        a, b: array-like, angles in radians
    
    Returns:
        diff: a - b, wrapped to [-pi, pi)
    """
    diff = a - b
    # Wrap to [-pi, pi)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def calculate_mse(gt_action_across_time, pred_action_across_time):
    T, D = gt_action_across_time.shape

    pos_indices = [0, 1, 2, 6, 7, 8]
    rpy_indices = [3, 4, 5, 9, 10, 11]

    # Position and gripper: still use Euclidean
    sq_error = gt_action_across_time - pred_action_across_time
    pos_l1_loss = np.mean(np.abs(sq_error[:, pos_indices]))
    hand_l1_loss = None
    hand_l1_loss = np.mean(np.abs(sq_error[:, 12:]))

    # RPY: use angular difference
    gt_rpy = gt_action_across_time[:, rpy_indices]      # (T, 6)
    pred_rpy = pred_action_across_time[:, rpy_indices]  # (T, 6)
    
    # Compute angular difference for each RPY component
    rpy_diff = angular_diff(gt_rpy, pred_rpy)           # (T, 6)
    rpy_l1_loss = np.mean(np.abs(rpy_diff))                    # scalar


    print(f"Position L1 Loss:     {pos_l1_loss:.6f}")
    print(f"Orientation L1 Loss:  {rpy_l1_loss:.6f}")
    if hand_l1_loss is not None:
        print(f"Hand L1 Loss:        {hand_l1_loss:.6f}")

    return {
        "position_l1_loss": float(pos_l1_loss),
        "orientation_l1_loss": float(rpy_l1_loss),
        "hand_l1_loss": float(hand_l1_loss) if D > 14 else None,
        "gripper_l1_loss": None
    }

def save_npy_file(gt_action_across_time, pred_action_across_time, states_across_time, save_plot_path, traj_id):
    parent_dir = os.path.join(save_plot_path, f"traj_{traj_id}")
    os.makedirs(parent_dir, exist_ok=True)
    save_gt_path = os.path.join(parent_dir, "gt_action_across_time.npy")
    np.save(save_gt_path, gt_action_across_time)
    # np.save(save_gt_path, gt_delta_action_across_time)
    save_pred_path = os.path.join(parent_dir, "pred_action_across_time.npy")
    np.save(save_pred_path, pred_action_across_time)
    # np.save(save_pred_path, pred_delta_action_across_time)
    save_states_path = os.path.join(parent_dir, "states_across_time.npy")
    np.save(save_states_path, states_across_time)
    print(f"gt_action_across_time saved to {save_gt_path}")
    print(f"pred_action_across_time saved to {save_pred_path}")
    print(f"states_across_time saved to {save_states_path}")
    return parent_dir

def calc_mse_for_single_trajectory(
    policy: baseframework,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
    create_trajectory_video=False,
    video_output_path=None,
    original_video_path=None,
    processor=None
):
    if dataset._metadata.embodiment_tag.value in WO_GRIPPER_DATASET:
        wo_gripper = True
    else:
        wo_gripper = False
    if dataset._metadata.embodiment_tag.value in ROBOCASA_DATASET:
        is_robocasa = True
    else:
        is_robocasa = False
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    gt_delta_action_across_time = []
    pred_delta_action_across_time = []
    return_state = False if policy.config.framework.action_model.state_dim is None else True
 
    for step_count in range(steps):
        data_point = None

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data_with_transform(traj_id, step_count, processor, policy.config.framework.qwenvl.base_vlm, return_state=return_state)
            print("inferencing at step: ", step_count)
            # data_point['action'][:-1, :6] = calculate_delta_eef(data_point['action'][:, :6])
            # data_point['action'][:-1, 6:12] = calculate_delta_eef(data_point['action'][:, 6:12])
            batch = collate_fn([data_point])

            action_chunk = policy.predict_action(batch)
            action_chunk = torch.from_numpy(action_chunk["normalized_actions"][0])
            if is_robocasa:
                modality_keys = [
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                ]
                action_chunk = dataset.transforms.unapply({"action.left_arm": action_chunk[:, :7],
                    "action.right_arm": action_chunk[:, 7:14],
                    "action.left_hand": action_chunk[:, 14:20],
                    "action.right_hand": action_chunk[:, 20:26],
                    "action.waist": action_chunk[:, 26:29]
                })
            elif wo_gripper:
                modality_keys = ["left_eef_position", "left_eef_rotation", 
                "right_eef_position", "right_eef_rotation", 
                "left_mano_hand_param", "right_mano_hand_param"]
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.right_eef_position": action_chunk[:, 6:9],
                    "action.right_eef_rotation": action_chunk[:, 9:12],
                    "action.left_mano_hand_param": action_chunk[:, 12:18],
                    "action.right_mano_hand_param": action_chunk[:, 18:24]})
            else:
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.left_gripper": action_chunk[:, 6:7],
                    "action.right_eef_position": action_chunk[:, 69:72],
                    "action.right_eef_rotation": action_chunk[:, 72:75],
                    "action.right_gripper": action_chunk[:, 75:76]})             
                    
            for j in range(action_horizon):
                if is_robocasa:
                    concat_pred_action = np.concatenate(
                        [action_chunk[f'action.{key}'][j] for key in modality_keys]
                    )
                    concat_gt_action = data_point["action"][j]
                elif wo_gripper:
                    concat_pred_action = np.concatenate(
                    [action_chunk[f'action.{key}'][j] for key in modality_keys]
                    ) 
                    concat_gt_action = data_point["action"][j][:24]
                else:
                    concat_pred_action = np.concatenate(
                    [action_chunk[f'action.{key}'][j] for key in modality_keys]
                    ) 
                    
                    concat_gt_action = data_point["action"][j+1][:14]
                pred_action_across_time.append(concat_pred_action)
                gt_action_across_time.append(concat_gt_action)

    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    # gt_delta_action_across_time = np.array(gt_delta_action_across_time)[:steps-1]
    # pred_delta_action_across_time = np.array(pred_delta_action_across_time)[:steps-1]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    states_list = []
    for step_count in range(steps):
        data_point = dataset.get_step_data_with_transform(traj_id, step_count, processor, policy.config.framework.qwenvl.base_vlm)
        state = torch.from_numpy(data_point["state"][0])
        states_list.append(state)
    states_across_time = np.array(states_list)
    extrinsic = data_point["extrinsic"] 
    intrinsic = data_point["intrinsic"]
    has_cam_param = extrinsic is not None

    parent_dir = save_npy_file(gt_action_across_time, pred_action_across_time, states_across_time, save_plot_path, traj_id)
    if is_robocasa:
        l1_loss = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
        l1_loss_dict = {
            'position_l1_loss': l1_loss,
            'orientation_l1_loss': l1_loss,
            'gripper_l1_loss': l1_loss,
            "hand_l1_loss": l1_loss
        }
    else:
        l1_loss_dict = calculate_mse(gt_action_across_time, pred_action_across_time)

    action_dim = gt_action_across_time.shape[1]
    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "l1_loss": l1_loss_dict['position_l1_loss'],
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        full_plot_path = os.path.join(parent_dir, "plot_traj.png")
        plot_trajectory(info, full_plot_path)
    
    if has_cam_param and create_trajectory_video and video_output_path and original_video_path:
        print("overlaying action trajectory on the original video...")
        
        print(f"state sequence shape: {states_across_time.shape}")
        # use the new function to create the trajectory video
        create_action_trajectory_video(
            original_video_path,
            states_across_time,
            pred_action_across_time,
            gt_action_across_time,
            video_output_path,
            extrinsic,
            intrinsic,
            action_horizon=action_horizon,
            interval=dataset.img_interval,
        )
    
    return l1_loss_dict

def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    l1_loss = info["l1_loss"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized L1 Loss: {l1_loss:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
 
