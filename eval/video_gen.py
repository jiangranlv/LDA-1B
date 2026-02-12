
from locale import normalize
import logging
import socket
import argparse
from LDA.model.framework.base_framework import baseframework
import torch, os
import pandas as pd
import numpy as np
import cv2
from omegaconf import OmegaConf
import json
from typing import Dict
from PIL import Image
import av
from sklearn.decomposition import PCA

IMG_MEAN = [0.485, 0.456, 0.406]

def read_video_pyav(video_path, resize_img):
    """
    使用 PyAV 读取视频，兼容 AV1/H.264 等格式
    返回: List[RGB PIL Image or np.ndarray]
    """
    frames = []
    container = av.open(video_path)
    
    for frame in container.decode(video=0):
        # 转为 RGB numpy array (H, W, C)
        img = frame.to_rgb().to_ndarray()
        if hasattr(resize_img, '__call__'):
            # 假设 resize_img 接受 numpy array (H, W, C) in [0,255], uint8, RGB
            resized = resize_img(img)
            frames.append(resized)
        else:
            frames.append(img)
    
    container.close()
    return frames

def visualize_dino(patch_features, img_path):
    patch_grid = patch_features
    H, W, D = patch_grid.shape
    patch_grid_flat = patch_grid.reshape(-1, D)  # [H*W, D]

    # Apply PCA to reduce to 3D (RGB) or 1D (grayscale)
    pca = PCA(n_components=3)
    patch_pca = pca.fit_transform(patch_grid_flat)

    # Normalize for visualization
    patch_pca -= patch_pca.min()
    patch_pca /= (patch_pca.max() - patch_pca.min())

    # Reshape to patch image grid
    pca_image = patch_pca.reshape(H, W, 3)

    # Upsample to original 224x224 for display

    orig_w, orig_h = (640, 480)
    pca_image = cv2.resize(
        pca_image,
        (orig_w, orig_h),           # 目标尺寸 (width, height)
        interpolation=cv2.INTER_LINEAR  # 双线性插值
    )
    pca_image = (np.clip(pca_image, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(pca_image)
    img.save(img_path)
    print(f"Img have been saved in {img_path}")

def read_jsonl_as_list(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def normalize_states(unnormalized_states: np.ndarray, state_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Args:
        normalized_actions: shape (B, chunk, D) (chunk, D)
        action_norm_stats:
    Returns:
        normalized_states
    """
    state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
    # if state_high == state_low, set the value to state_high
    mask = state_high == state_low
    normalized_states = (unnormalized_states - state_low) / (state_high - state_low + 1e-8) * 2 - 1
    normalized_states = np.clip(normalized_states, -1, 1)
    normalized_states[..., mask] = 0
    
    return normalized_states

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_img(rgb_obs):
    background_color = tuple(int(x*255) for x in IMG_MEAN)
    image = Image.fromarray(rgb_obs)
    image = expand2square(image, background_color)
    image = image.resize((224, 224))
    return image

def prepare_inputs(state_stats, data_path, video_path, embodiment_id, tasks):

    # === 1. 从 Parquet 读取元数据 ===
    df = pd.read_parquet(data_path)

    task_index = df['task_index']
    lang = [tasks[index]['task'] for index in task_index]
    state = None
    if embodiment_id == 27:
        state = np.stack(df["observation.state"])
        state = normalize_states(state, state_stats)

    history_action = None
    if "history_action" in df and not pd.isna(df["history_action"]):
        history_action = np.array(df["history_action"], dtype=np.float32)  # [T_hist, action_dim]

    # === 2. 从视频加载帧 ===
    frames = read_video_pyav(video_path, resize_img)

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")
    # === 3. 构造 example ===
    examples = []
    for i in range(len(frames)):
        if i < len(lang):
            language = str(lang[i])
        else:
            language = str(lang[0]) # for intern_genie1, intern genie1 被抽帧了，所以lang的长度时frames的 1 / 3
        example = { 
            "image": [frames[i]],  
            "lang": language,
            "embodiment_id": int(embodiment_id),
        }
        if state is not None:
            example["state"] = state[i][np.newaxis, :]
        if history_action is not None:
            example["history_action"] = history_action
        examples.append(example)

    return examples
        
    
def main(args) -> None:

    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )

    if args.use_bf16: 
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    stats_path = args.config_path.replace('config.yaml', 'dataset_statistics.json')
    with open(stats_path, "r") as f:
        stats = json.load(f)
    state_stats = stats[args.robot_name]["state"]
    action_stats = stats[args.robot_name]["action"]

    tasks_path = args.tasks_path
    tasks = read_jsonl_as_list(tasks_path)

    examples = prepare_inputs(state_stats, args.data_path, args.video_path, args.embodiment_id, tasks)

    all_pred_obs = []
    os.makedirs(args.save_path, exist_ok=True)
    for i in range(len(examples)):
        pred_obs = vla.video_gen(examples[i])
        img_patch = pred_obs["normalized_obs"][0][5:].reshape(40, 30, -1)
        all_pred_obs.append(pred_obs["normalized_obs"][0])
        visualize_dino(img_patch, args.save_path + f"/{i}.png")
    all_pred_obs = np.array(all_pred_obs)
    
    np.save(args.save_path + f"/{i}.npy", all_pred_obs)

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--tasks_path", type=str)
    parser.add_argument("--save_path", type=str, help="path to save the visulization result")
    parser.add_argument("--embodiment_id", type=int, default=27)
    parser.add_argument("--robot_name", type=str, default="galbot")
    parser.add_argument("--use_bf16", action="store_true")
    return parser

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)