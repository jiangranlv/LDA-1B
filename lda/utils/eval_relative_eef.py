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

from LDA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from LDA.model.framework.base_framework import baseframework
import h5py
import torch
from pytorch3d.transforms import euler_angles_to_matrix
# numpy print precision settings 3, dont use exponential notation

from LDA.utils.rotation_convert import delta2abs
from LDA.dataloader.lerobot_datasets import collate_fn

AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
PIXEL_EPS = 1e-9
PRED_COLOR_BASE = {"leftHand": (0, 0, 255), "rightHand": (0, 0, 255)} # 
GT_COLOR_BASE = {"leftHand": (255, 255, 0), "rightHand": (255, 255, 0)} # 
AXIS_LENGTH = 0.1
AXIS_THICKNESS = 2
POINT_RADIUS = 5

WO_GRIPPER_DATASET = ['human', 'egovla', 'ssv2', 'robocasa']
MANO_HAND_DATASET = ['egovla', 'ssv2']

np.set_printoptions(precision=3, suppress=True)

def to_pose(trans, rot_mat):
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T

def _pixel_to_cv(pixel: np.ndarray) -> tuple[int, int]:
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def add_pose_axes(image: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray,
                  axis_length: float, thickness: int) -> np.ndarray:
    origin = pose[:3, 3]
    origin_px = point_3d_to_2d(intrinsics, origin)

    for axis_idx in range(3):
        axis_tip = origin + axis_length * pose[:3, axis_idx]
        tip_px = point_3d_to_2d(intrinsics, axis_tip)

        cv2.line(
            image,
            _pixel_to_cv(origin_px),
            _pixel_to_cv(tip_px),
            AXIS_COLORS[axis_idx],
            thickness
        )

        z = pose[2, axis_idx]

        tip_px_cv = _pixel_to_cv(tip_px)

        if z > 0:
            cv2.circle(image, tip_px_cv, radius=5, color=AXIS_COLORS[axis_idx], thickness=-1)
        else:
            cv2.circle(image, tip_px_cv, radius=6, color=AXIS_COLORS[axis_idx], thickness=2)

    return image

def add_point(image: np.ndarray, pixel: np.ndarray, color: tuple[int, int, int], radius: int) -> np.ndarray:
    cv2.circle(image, _pixel_to_cv(pixel), radius, color, -1)
    return image

def add_line(image: np.ndarray, pixel1: np.ndarray, pixel2: np.ndarray, color: tuple[int, int, int], thickness: int = AXIS_THICKNESS) -> np.ndarray:
    cv2.line(image, _pixel_to_cv(pixel1), _pixel_to_cv(pixel2), color, thickness)
    return image

def point_3d_to_2d(intrinsics: np.ndarray, point: np.ndarray) -> np.ndarray:
    orig_shape = point.shape
    coords = point.reshape(-1, 3).astype(np.float32)
    depth = coords[:, [2]]
    depth_sign = np.where(depth >= 0.0, 1.0, -1.0).astype(np.float32)
    depth = np.where(np.abs(depth) < PIXEL_EPS, depth_sign * PIXEL_EPS, depth)
    norm = np.concatenate([coords[:, :2] / depth, np.ones_like(depth)], axis=1)
    pixels = np.einsum("ab,nb->na", intrinsics, norm, dtype=np.float32)
    return pixels.reshape(*orig_shape[:-1], 3)[..., :2]


def project_eef_pose_to_image(left_eef, right_eef, cam, int_param, frame, past_left_eef=None, past_right_eef=None, point_color=None, gt=False, add_axes=False):
    """
    left_eef: (4, 4)
    right_eef: (4, 4)
    cam: (16,)
    int_param: (9,)
    frame: cv2 image
    gt: bool, default False
    return: cv2 image
        - image: (H, W, 3)
    """
    left_point_color = point_color["leftHand"]
    right_point_color = point_color["rightHand"]
    cam_inv = np.linalg.inv(cam.reshape(4, 4))
    left_eef_in_cam = cam_inv @ left_eef
    right_eef_in_cam = cam_inv @ right_eef 

    if past_left_eef is not None:
        past_left_eef_in_cam = cam_inv @ past_left_eef
        past_right_eef_in_cam = cam_inv @ past_right_eef

    int_param = int_param.reshape(3, 3).copy()
    # 2d point in image
    left_2d_point = point_3d_to_2d(int_param, left_eef_in_cam[:3, 3])
    right_2d_point = point_3d_to_2d(int_param, right_eef_in_cam[:3, 3])
    if past_left_eef is not None:
        past_left_2d_point = point_3d_to_2d(int_param, past_left_eef_in_cam[:3, 3])
        past_right_2d_point = point_3d_to_2d(int_param, past_right_eef_in_cam[:3, 3])

    img = frame.copy()
    if add_axes:
        img = add_pose_axes(img, left_eef_in_cam, int_param, AXIS_LENGTH, AXIS_THICKNESS)
        img = add_pose_axes(img, right_eef_in_cam, int_param, AXIS_LENGTH, AXIS_THICKNESS)
    img = add_point(img, left_2d_point, left_point_color, POINT_RADIUS)
    img = add_point(img, right_2d_point, right_point_color, POINT_RADIUS)

    if past_left_eef is not None:
        img = add_line(img, past_left_2d_point, left_2d_point, left_point_color, AXIS_THICKNESS)
        img = add_line(img, past_right_2d_point, right_2d_point, right_point_color, AXIS_THICKNESS)

    cv2.putText(img, "GT: Yellow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img, "Pred: Blue", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img

    
def convert_euler_to_matrix(euler_angles, convention='XYZ'):
    """
    convert euler angles to rotation matrix
    euler_angles: [x, y, z] euler angles (radians)
    convention: euler angle order, default is 'XYZ'
    """
    euler_tensor = torch.tensor(euler_angles, dtype=torch.float32).unsqueeze(0)
    rotation_matrix = euler_angles_to_matrix(euler_tensor, convention)
    return rotation_matrix.squeeze(0).numpy()

def compute_future_trajectory(current_pos, current_rot_mat, current_euler, actions, horizon, is_arm2=False):
    """
    compute the future trajectory from the current position
    
    参数:
    - current_pos: [x, y, z]
    - current_rot_mat: rotation matrix (3x3)
    - current_rotation_6d: 6D rotation [a1, a2, a3, b1, b2, b3]
    - actions: action sequence
    - horizon: prediction steps
    - is_arm2: whether is the second arm
    """
    trajectory = np.zeros((horizon + 1, 3))
    rotations = [current_rot_mat.copy()]
    trajectory[0] = current_pos  # start from the current position
    
    current_pos = current_pos.copy()
    current_rot = current_rot_mat.copy()
    
    for i in range(min(horizon, len(actions))):
        if is_arm2:
            current_pos = actions[i, 6:9]
            
            if np.any(actions[i, 9:12] != 0):
                current_euler = actions[i, 9:12]
                current_rot = convert_euler_to_matrix(current_euler)
        else:
            current_pos = actions[i, :3]
            
            if np.any(actions[i, 3:6] != 0):
                current_euler = actions[i, 3:6]
                current_rot = convert_euler_to_matrix(current_euler)
        
        trajectory[i + 1] = current_pos
        rotations.append(current_rot.copy())
    
    return trajectory, rotations

def create_action_trajectory_video(
    original_video_path,
    states_across_time,
    pred_actions_across_time,
    gt_actions_across_time,
    output_video_path,
    extrinsic,
    intrinsic,
    action_horizon=16,
    interval=3,
):
    """
    project the trajectory on the original video
    """
    
    # load camera parameters
    int_param = intrinsic.to_numpy()
    extrinsic_params_list = extrinsic.to_numpy()
    
    steps = len(states_across_time)
    
    print(f"trajectory data information:")
    print(f"  state sequence shape: {states_across_time.shape}")
    print(f"  predicted action shape: {pred_actions_across_time.shape}")
    print(f"  ground truth action shape: {gt_actions_across_time.shape}")
    print(f"  extrinsic: {extrinsic.shape}")
    print(f"  intrinsic: {intrinsic.shape}")
            
    # relative end-effector position action processing
    current_arm1_positions = states_across_time[:, 0:3]  # (steps, 3)
    current_arm2_positions = states_across_time[:, 6:9]  # (steps, 3)

    current_arm1_euler_all = states_across_time[:, 3:6]  # (steps, 3) - euler angles
    current_arm2_euler_all = states_across_time[:, 9:12]  # (steps, 3) - euler angles
    
    print(f"  Arm1 position range: X[{current_arm1_positions[:,0].min():.3f}, {current_arm1_positions[:,0].max():.3f}], "
            f"Y[{current_arm1_positions[:,1].min():.3f}, {current_arm1_positions[:,1].max():.3f}], "
            f"Z[{current_arm1_positions[:,2].min():.3f}, {current_arm1_positions[:,2].max():.3f}]")
    # open the original video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"cannot open the original video: {original_video_path}")
        return
    
    # get the video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"original video: {frame_width}x{frame_height}, {original_fps}fps, {total_frames} frames")
    
    frame_count = 0
    valid_frames = 0

    imgs = []
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 3 * steps:
            break
        if frame_count % interval != 0:
            frame_count += 1
            continue
            
        # get the camera extrinsic parameters for the current frame
        if valid_frames < len(extrinsic_params_list):
            index = valid_frames
        else:
            index = -1
            
        # use the end-effector position and rotation in the state
        current_arm1_pos = current_arm1_positions[valid_frames]
        current_arm2_pos = current_arm2_positions[valid_frames]
        current_arm1_euler = current_arm1_euler_all[valid_frames]
        current_arm2_euler = current_arm2_euler_all[valid_frames]
        
        # convert the 6D rotation to a rotation matrix
        current_arm1_rot = convert_euler_to_matrix(current_arm1_euler)
        current_arm2_rot = convert_euler_to_matrix(current_arm2_euler)
        
        # compute the current ground truth end-effector pose
        left_eef_log = to_pose(current_arm1_pos, current_arm1_rot)
        right_eef_log = to_pose(current_arm2_pos, current_arm2_rot)
        
        # calculate current eef position
        if valid_frames < len(pred_actions_across_time):
                
            pred_arm1_pos = pred_actions_across_time[valid_frames, :3]
            pred_arm2_pos = pred_actions_across_time[valid_frames, 6:9]

            pred_arm1_euler = pred_actions_across_time[valid_frames, 3:6]
            pred_arm2_euler = pred_actions_across_time[valid_frames, 9:12]
            
            pred_arm1_rot = convert_euler_to_matrix(pred_arm1_euler)
            pred_arm2_rot = convert_euler_to_matrix(pred_arm2_euler)
            
            # calculate predicted eef pose
            pred_left_eef_log = to_pose(pred_arm1_pos, pred_arm1_rot)
            pred_right_eef_log = to_pose(pred_arm2_pos, pred_arm2_rot)
        
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. draw the current ground truth eef pose
        img = project_eef_pose_to_image(left_eef_log, right_eef_log, extrinsic_params_list[index], int_param[index], img, point_color=GT_COLOR_BASE, gt=True)
        
        # 3. draw the current predicted eef point
        if valid_frames < len(pred_actions_across_time):
            img = project_eef_pose_to_image(pred_left_eef_log, pred_right_eef_log, extrinsic_params_list[index], int_param[index], img, point_color=PRED_COLOR_BASE, gt=False)
        
        # 4. calculate and draw the future 16 frame trajectory
        if valid_frames < steps - 1:
            # get the future action sequence
            future_start = valid_frames
            future_end = min(valid_frames + action_horizon, len(pred_actions_across_time))
            
            future_pred_actions = pred_actions_across_time[future_start:future_end]
            future_gt_actions = gt_actions_across_time[future_start:future_end]
            
            # if the action sequence is not long enough, pad with zeros
            if len(future_pred_actions) < action_horizon:
                padding_size = action_horizon - len(future_pred_actions)
                if len(future_pred_actions) > 0:
                    padding = np.zeros((padding_size, future_pred_actions.shape[1]))
                    future_pred_actions = np.vstack([future_pred_actions, padding])
                    future_gt_actions = np.vstack([future_gt_actions, padding])
                else:
                    future_pred_actions = np.zeros((action_horizon, pred_actions_across_time.shape[1]))
                    future_gt_actions = np.zeros((action_horizon, gt_actions_across_time.shape[1]))
            
            # calculate the future trajectory of the relative end-effector position action
            future_gt_arm1, future_gt_arm1_rots = compute_future_trajectory(
                current_arm1_pos, current_arm1_rot, current_arm1_euler,
                future_gt_actions, action_horizon, False)
            future_gt_arm2, future_gt_arm2_rots = compute_future_trajectory(
                current_arm2_pos, current_arm2_rot, current_arm2_euler,
                future_gt_actions, action_horizon, True)
            future_pred_arm1, future_pred_arm1_rots = compute_future_trajectory(
                current_arm1_pos, current_arm1_rot, current_arm1_euler,
                future_pred_actions, action_horizon, False)
            future_pred_arm2, future_pred_arm2_rots = compute_future_trajectory(
                current_arm2_pos, current_arm2_rot, current_arm2_euler,
                future_pred_actions, action_horizon, True)
            
            past_left_eef = left_eef_log
            past_right_eef = right_eef_log
            past_pred_left_eef = pred_left_eef_log
            past_pred_right_eef = pred_right_eef_log
            for i in range(len(future_gt_arm1)):
                alpha = 1.0 - (i / max(action_horizon - 1, 1))  # normalize to [0, 1], i=0 → alpha=1.0

                # map alpha to color brightness (keep saturation, decrease brightness)
                # method: color * alpha + white * (1 - alpha) → mix with white (lighten)
                def blend_to_white(color, alpha):
                    color = np.array(color, dtype=np.float32)
                    white = np.array([255, 255, 255], dtype=np.float32)
                    blended = alpha * color + (1 - alpha) * white
                    return tuple(blended.astype(np.uint8))

                gt_color = {k: blend_to_white(GT_COLOR_BASE[k], alpha) for k in GT_COLOR_BASE.keys()}
                pred_color = {k: blend_to_white(PRED_COLOR_BASE[k], alpha) for k in PRED_COLOR_BASE.keys()}
                gt_color = GT_COLOR_BASE
                pred_color = PRED_COLOR_BASE

                future_frame_idx = valid_frames + i
                if future_frame_idx < len(extrinsic_params_list):
                    pass
                else:
                    future_frame_idx = -1
                
                # build the pose matrix for the relative end-effector position action
                gt_arm1_pose = to_pose(future_gt_arm1[i], future_gt_arm1_rots[i])
                gt_arm2_pose = to_pose(future_gt_arm2[i], future_gt_arm2_rots[i])
                
                # predicted trajectory point
                pred_arm1_pose = to_pose(future_pred_arm1[i], future_pred_arm1_rots[i])
                pred_arm2_pose = to_pose(future_pred_arm2[i], future_pred_arm2_rots[i])

                img = project_eef_pose_to_image(gt_arm1_pose, gt_arm2_pose, extrinsic_params_list[future_frame_idx], int_param[future_frame_idx], img, past_left_eef=past_left_eef, past_right_eef=past_right_eef, point_color=gt_color, gt=True)
                img = project_eef_pose_to_image(pred_arm1_pose, pred_arm2_pose, extrinsic_params_list[future_frame_idx], int_param[future_frame_idx], img, past_left_eef=past_pred_left_eef, past_right_eef=past_pred_right_eef, point_color=pred_color, gt=False)

                past_left_eef = gt_arm1_pose
                past_right_eef = gt_arm2_pose
                past_pred_left_eef = pred_arm1_pose
                past_pred_right_eef = pred_arm2_pose
        
        valid_frames += 1
    
        imgs.append(img)
        frame_count += 1

    cap.release()
    # cv2.destroyAllWindows()
    imageio.mimwrite(output_video_path, imgs, fps=original_fps // interval, codec='libx264')
    
    print(f"action trajectory video saved to: {output_video_path}")
    print(f"valid frames (with trajectory displayed): {valid_frames}/{frame_count}")
    
    return valid_frames > 0

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
    if D == 14:
        pos_indices = [0, 1, 2, 7, 8, 9]
        rpy_indices = [3, 4, 5, 10, 11, 12]
        gripper_indices = [6, 13]
    else:
        pos_indices = [0, 1, 2, 6, 7, 8]
        rpy_indices = [3, 4, 5, 9, 10, 11]

    # Position and gripper: still use Euclidean
    sq_error = gt_action_across_time - pred_action_across_time
    pos_l1_loss = np.mean(np.abs(sq_error[:, pos_indices]))
    gripper_l1_loss = None
    hand_l1_loss = None
    if D == 14:
        gripper_l1_loss = np.mean(np.abs(sq_error[:, gripper_indices]))
    elif D > 14:
        hand_l1_loss = np.mean(np.abs(sq_error[:, 12:]))

    # RPY: use angular difference
    gt_rpy = gt_action_across_time[:, rpy_indices]      # (T, 6)
    pred_rpy = pred_action_across_time[:, rpy_indices]  # (T, 6)
    
    # Compute angular difference for each RPY component
    rpy_diff = angular_diff(gt_rpy, pred_rpy)           # (T, 6)
    rpy_l1_loss = np.mean(np.abs(rpy_diff))                    # scalar


    print(f"Position L1 Loss:     {pos_l1_loss:.6f}")
    print(f"Orientation L1 Loss:  {rpy_l1_loss:.6f}")
    if gripper_l1_loss is not None:
        print(f"Gripper L1 Loss:      {gripper_l1_loss:.6f}")
    if hand_l1_loss is not None:
        print(f"Hand L1 Loss:        {hand_l1_loss:.6f}")

    return {
        "position_l1_loss": float(pos_l1_loss),
        "orientation_l1_loss": float(rpy_l1_loss),
        "gripper_l1_loss": float(gripper_l1_loss) if D == 14 else None,
        "hand_l1_loss": float(hand_l1_loss) if D > 14 else None,
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
):
    wo_gripper = False # default is robot dataset, with gripper, no mano hand
    wo_mano = True
    if dataset._metadata.embodiment_tag.value in WO_GRIPPER_DATASET:
        wo_gripper = True
        if dataset._metadata.embodiment_tag.value in MANO_HAND_DATASET:
            wo_mano = False
    if dataset._metadata.embodiment_tag.value == "robocasa":
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
                data_point = dataset.get_step_data_with_transform(traj_id, step_count, None, policy.config.framework.qwenvl.base_vlm, return_state=return_state)

            print("inferencing at step: ", step_count)
            batch = collate_fn([data_point])
            action_chunk = policy.predict_action(batch)
            action_chunk = torch.from_numpy(action_chunk["normalized_actions"][0])

            if is_robocasa:
                modality_keys = ["left_eef_position", "left_eef_rotation", 
                "right_eef_position", "right_eef_rotation",
                "left_mano_hand", "right_mano_hand"]
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.right_eef_position": action_chunk[:, 6:9],
                    "action.right_eef_rotation": action_chunk[:, 9:12],
                    "action.left_mano_hand": action_chunk[:, 12:18],
                    "action.right_mano_hand": action_chunk[:, 75:81]})
            elif wo_gripper and (not wo_mano):
                modality_keys = ["left_eef_position", "left_eef_rotation", 
                "right_eef_position", "right_eef_rotation",
                "left_mano_hand_param", "right_mano_hand_param"]
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.right_eef_position": action_chunk[:, 6:9],
                    "action.right_eef_rotation": action_chunk[:, 9:12],
                    "action.left_mano_hand": action_chunk[:, 12:75],
                    "action.right_mano_hand": action_chunk[:, 75:138]})
            elif wo_gripper and wo_mano:
                modality_keys = ["left_eef_position", "left_eef_rotation", 
                "right_eef_position", "right_eef_rotation"]
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.right_eef_position": action_chunk[:, 6:9],
                    "action.right_eef_rotation": action_chunk[:, 9:12]})
            else:
                modality_keys = ["left_eef_position", "left_eef_rotation", "action.left_gripper",
                "right_eef_position", "right_eef_rotation", "action.right_gripper",]
                action_chunk = dataset.transforms.unapply({"action.left_eef_position": action_chunk[:, :3],
                    "action.left_eef_rotation": action_chunk[:, 3:6],
                    "action.left_gripper": action_chunk[:, 6:7],
                    "action.right_eef_position": action_chunk[:, 69:72],
                    "action.right_eef_rotation": action_chunk[:, 72:75],
                    "action.right_gripper": action_chunk[:, 75:76]})
                left_action_abs_pose, right_action_abs_pose = get_abs_eef(action_chunk, data_point, wo_gripper)
                if not wo_gripper:
                    # if dataset._metadata.embodiment_tag.value == "agibot_alpha":
                    #     action_chunk["action.left_gripper"] = 1 - action_chunk["action.left_gripper"]
                    #     action_chunk["action.right_gripper"] = 1 - action_chunk["action.right_gripper"]
                    left_action_gripper = np.concatenate([data_point["action"][0:1, 6:7], 1 - action_chunk["action.left_gripper"]])
                    right_action_gripper = np.concatenate([data_point["action"][0:1, 13:14], 1 - action_chunk["action.right_gripper"]])
                    
            for j in range(action_horizon):
                # predict action
                if wo_gripper and wo_mano:
                    pred_action_across_time.append(np.concatenate([left_action_abs_pose[j],
                     right_action_abs_pose[j]]))
                    concat_gt_action = data_point["action"][j][:12]
                elif wo_gripper and (not wo_mano):
                    pred_action_across_time.append(np.concatenate([left_action_abs_pose[j],
                     right_action_abs_pose[j], action_chunk["action.left_mano_hand"][j],
                     action_chunk["action.right_mano_hand"][j]]))
                    concat_gt_action = data_point["action"][j][:138]
                else:
                    pred_action_across_time.append(np.concatenate([left_action_abs_pose[j], left_action_gripper[j], right_action_abs_pose[j], right_action_gripper[j]]))
                    concat_gt_action = data_point["action"][j][:14]

                gt_action_across_time.append(concat_gt_action)

    # convert to numpy array
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    # gt_delta_action_across_time = np.array(gt_delta_action_across_time)[:steps-1]
    # pred_delta_action_across_time = np.array(pred_delta_action_across_time)[:steps-1]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    states_list = []
    for step_count in range(steps):
        data_point = dataset.get_step_data_with_transform(traj_id, step_count, None, policy.config.framework.qwenvl.base_vlm)
        state = torch.from_numpy(data_point["state"][0])
        states_list.append(state)
    states_across_time = np.array(states_list)
    extrinsic = data_point["extrinsic"] 
    intrinsic = data_point["intrinsic"]
    has_cam_param = extrinsic is not None

    # save gt_action_across_time and pred_action_across_time to file
    parent_dir = save_npy_file(gt_action_across_time, pred_action_across_time, states_across_time, save_plot_path, traj_id)
    # calculate MSE besides gripper
    mse_dict = calculate_mse(gt_action_across_time, pred_action_across_time)

    action_dim = gt_action_across_time.shape[1]
    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "position_l1_loss": mse_dict["position_l1_loss"],
            "orientation_l1_loss": mse_dict["orientation_l1_loss"],
            "gripper_l1_loss": mse_dict["gripper_l1_loss"],
            "hand_l1_loss": mse_dict["hand_l1_loss"],
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
    
    return mse_dict

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
    l1_loss = info["position_l1_loss"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized L1 Loss: {l1_loss:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
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

if __name__ == "__main__":
    # debug
    gt_path = "/mnt/project/world_model/checkpoints/gr00t_agibot_egodex_galaxea_lr1e-4_w_state_w_vlm_attn_mask_DiT-L/results/steps_150000/agibot_alpha/traj_6/gt_action_across_time.npy"
    pred_path = "/mnt/project/world_model/checkpoints/gr00t_agibot_egodex_galaxea_lr1e-4_w_state_w_vlm_attn_mask_DiT-L/results/steps_150000/agibot_alpha/traj_6/pred_action_across_time.npy"
    gt_action_across_time = np.load(gt_path)
    pred_action_across_time = np.load(pred_path)
    calculate_mse(gt_action_across_time, pred_action_across_time)
    # create_action_trajectory_video(
    #     original_video_path=original_video_path,
    #     states_across_time=states_across_time,
    #     pred_actions_across_time=pred_action_across_time,
    #     gt_actions_across_time=gt_action_across_time,
    #     output_video_path=output_video_path,
    #     extrinsic_path=extrinsic_path,
    #     intrinsic_path=intrinsic_path,
    # )
