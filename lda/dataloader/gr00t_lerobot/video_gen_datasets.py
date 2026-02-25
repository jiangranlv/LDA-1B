import os
import glob
import numpy as np
import cv2
from PIL import Image
import json
import pickle
from pathlib import Path
from torch.utils.data import Dataset

from lda.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps
from lda.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from lda.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from lda.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

def build_metadata_cache(video_paths, cache_path="metadata.pkl"):
    metadata = {}
    for vp in video_paths:
        json_path = os.path.splitext(vp)[0] + ".json"
        if not os.path.exists(json_path):
            raise RuntimeError(f"Missing {json_path}")
        with open(json_path, "r") as f:
            info = json.load(f)
        metadata[vp] = info  # or just store needed fields: fps, duration_sec
    with open(cache_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata for {len(metadata)} videos to {cache_path}")

class VideoTaskSingleDataset(Dataset):
    """
    Compatible replacement for LeRobotSingleDataset, only uses video + task description.
    Other modalities are zero placeholders.
    """
    def __init__(
        self,
        trajectory_root: str,
        modality_configs: dict,
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        transforms=None,
        task_descriptions=None,
        target_fps: int = 10,
        metadata_cache_path: str = None,
        **kwargs,
    ):
        self.trajectory_root = trajectory_root
        self.modality_configs = modality_configs
        self._embodiment_tag = embodiment_tag
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag
        self.video_backend = video_backend
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )
        self.target_fps = target_fps

        # 视频 + task
        all_video_paths = sorted(glob.glob(os.path.join(trajectory_root, "**", "*.mp4"), recursive=True))

        self.video_paths = []
        for vp in all_video_paths:
            json_path = os.path.splitext(vp)[0] + ".json"
            if os.path.exists(json_path):
                self.video_paths.append(vp)
        if task_descriptions is None: 
            self.task_descriptions = ["work in the factory."] * len(self.video_paths)
        assert len(self.video_paths) == len(self.task_descriptions), "Videos and tasks must match"

        if metadata_cache_path and os.path.exists(metadata_cache_path):
            pass
        else:
            build_metadata_cache(self.video_paths, metadata_cache_path)
        with open(metadata_cache_path, "rb") as f:
            all_metadata = pickle.load(f)
        # 确保顺序一致
        metadata_list = [all_metadata[vp] for vp in self.video_paths]

        # timestamps & trajectory_lengths
        self.timestamps = []
        self.trajectory_lengths = []
        for video_path, info in zip(self.video_paths, metadata_list):
            # 对应的 JSON 文件
            # json_path = os.path.splitext(video_path)[0] + ".json"
            # if not os.path.exists(json_path):
            #     raise RuntimeError(f"Cannot find json file {json_path} for video {video_path}")

            # with open(json_path, "r") as f:
            #     info = json.load(f)

            fps = info.get("fps", 30.0)
            duration = info.get("duration_sec", None)
            if duration is not None:
                n_frames = int(round(fps * duration))
            else:
                n_frames = int(info.get("size_bytes", 0) / 1000000)  # fallback
            stride = max(1, int(round(fps / self.target_fps)))
            frame_indices = np.arange(0, n_frames, stride)
            ts = frame_indices / fps
            self.timestamps.append(ts)
            self.trajectory_lengths.append(len(ts))
        self.trajectory_lengths = np.array(self.trajectory_lengths)

        # 轨迹 id
        self.all_steps = []
        self.trajectory_ids = list(range(len(self.video_paths)))
        for trajectory_id in self.trajectory_ids:
            for step in range(self.trajectory_lengths[trajectory_id]):
                self.all_steps.append((trajectory_id, step))
        self.all_steps = np.array(self.all_steps)

        self.history_action_indices = kwargs.get("history_action_indices", None)

        self._metadata = self._get_metadata()
        self.delta_indices = self._get_delta_indices()
        self.modality_keys = self._get_modality_keys()

    def _get_metadata(self):
        """
        Minimal metadata, compatible with LeRobotSingleDataset
        """
        class Metadata:
            pass

        metadata = Metadata()
        metadata.embodiment_tag = self._embodiment_tag
        return metadata

    def get_video_path(self, trajectory_id, key):
        video_path = self.video_paths[trajectory_id]
        return video_path

    def _get_modality_keys(self):
        modality_keys = {}
        for key, value in self.modality_configs.items():
            modality_keys[key] = value.modality_keys
        return modality_keys

    def _get_delta_indices(self):
        delta_indices = {}
        for key, value in self.modality_configs.items():
            delta_indices[key] = np.array(value.delta_indices)
        return delta_indices

    def __len__(self):
        return len(self.all_steps)

    def get_video_frames(self, video_path, ts):
        if len(ts) == 0:
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)
        frames = get_frames_by_timestamps(video_path, ts)
        return frames

    def restric_timestamps(self, step_indices, trajectory_id):
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_id] - 1)
        return step_indices

    def get_step_data(self, trajectory_id, step):
        step_indices = self.delta_indices["video"] + step 
        step_indices = self.restric_timestamps(step_indices, trajectory_id)
        ts = self.timestamps[trajectory_id][step_indices]
        n_frames = len(ts)
        data = {}
        # 视频
        for video_key in self.modality_keys["video"]:
            data[video_key] = self.get_video_frames(self.video_paths[trajectory_id], ts)
        
        # 未来帧
        future_step_indices = self.delta_indices["future_video"] + step
        future_step_indices = self.restric_timestamps(future_step_indices, trajectory_id)
        future_ts = self.timestamps[trajectory_id][future_step_indices]
        for future_video_key in self.modality_keys["future_video"]:
            data[future_video_key] = self.get_video_frames(self.video_paths[trajectory_id], future_ts)

        # 语言
        for lang_key in self.modality_keys.get("language", []):
            data[lang_key] = [self.task_descriptions[trajectory_id]]

        # 其他模态占位
        for action_key in self.modality_keys.get("action", []):
            data[action_key] = np.zeros((len(self.delta_indices['action']), 1), dtype=np.float32)
        for state_key in self.modality_keys.get("state", []):
            data[state_key] = np.zeros((len(self.delta_indices['state']), 1), dtype=np.float32)
        for history_action_key in self.modality_keys.get("history_action", []):
            data[history_action_key] = np.zeros((len(self.history_action_indices), 1), dtype=np.float32)

        return data

    def __getitem__(self, index):
        trajectory_id, step = self.all_steps[index]
        data = self.get_step_data(trajectory_id, step)
        return data


if __name__ == "__main__":
    # debug code
    from lda.dataloader.gr00t_lerobot.data_config import EgoCentric10KDataConfig
    from lda.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    data_config = EgoCentric10KDataConfig()
    modality_configs = data_config.modality_config()
    embodiment_tag = EmbodimentTag.EGOCENTRIC_10K
    video_backend = data_config.video_backend
    transforms = data_config.transform()
    target_fps = data_config.target_fps
    dataset = VideoTaskSingleDataset(trajectory_root="/mnt/project/public/world_model/RawData/egocentric-10k/egocentric-10k_extracted", 
    modality_configs=modality_configs, embodiment_tag=embodiment_tag, video_backend=video_backend, transforms=transforms, target_fps=target_fps)
    
    mixture_dataset = LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0)],
        mode="train",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=42,
        use_state=False,
        metadata_config={
            "percentile_mixing_method": "min_max",
        },
    )