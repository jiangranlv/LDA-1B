# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from lda.dataloader.gr00t_lerobot.datasets import ModalityConfig
from lda.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from lda.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from lda.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from lda.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    video_backend = "torchvision_av"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.left_gripper",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_gripper",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.right_gripper",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0))
    action_indices = list(range(-5, 17))
    img_interval = 3


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_position": "q99", 
                    "action.right_eef_position": "q99",
                    "action.left_eef_rotation": "q99",
                    "action.right_eef_rotation": "q99",
                    "action.left_gripper": "q99",
                    "action.right_gripper": "q99",
                    },
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class HumanBaseDataConfig(ABC):
    video_backend = "decord"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.left_mano_hand",
        "state.right_mano_hand",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_mano_hand",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.right_mano_hand",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0)) # indicate which part is history action
    action_indices = list(range(-5, 17))
    img_interval = 3


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)
###########################################################################################

class FourierGr1ArmsWaist_twohistoryDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [-5, 0]
    future_observation_indices = [5]
    history_action_indices = list(range(-5, 0))
    action_indices = list(range(-5, 16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    future_observation_indices = [16]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################
class FourierGr1EEFDataConfig:
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [-5, 0]
    history_action_indices = list(range(-5, 0))
    future_observation_indices = [5]
    action_indices = list(range(-5, 17))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

# RobotDataset

class AgibotWorldDataConfig(BaseDataConfig):
    pass

class AgibotDexDataConfig(BaseDataConfig):
    video_backend = "torchvision_av"

class GalaxeaDataConfig(BaseDataConfig):
    img_interval = 5
    video_backend = "torchvision_av"
    pass

class DroidDataConfig(BaseDataConfig):
    pass

class HumanoidEverydayDataConfig(BaseDataConfig):
    pass

class InternDataConfig(BaseDataConfig):
    pass

class FrankaDataConfig(BaseDataConfig):
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.left_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_gripper",
    ]
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_position": "q99", 
                    "action.left_eef_rotation": "q99",
                    "action.left_gripper": "binary",
                    },
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class OxeDataConfig(BaseDataConfig):
    pass

class RoboCoin_g1eduDataConfig(BaseDataConfig):
    video_backend = "torchvision_av"
    pass

class RoboCoin_lejuDataConfig(BaseDataConfig):
    pass

class RoboCoin_r1liteDataConfig(BaseDataConfig):
    pass

class RobomindDataConfig(BaseDataConfig):
    pass

class Challange2025DataConfig(BaseDataConfig):
    pass

class RH20TDataConfig(BaseDataConfig):
    pass

# Human Data Config
class VitraDataConfig(HumanBaseDataConfig):
    pass

class EgodexDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
    ]

class Hoi4dDataConfig(HumanBaseDataConfig):
    pass

class HoloAssitDataConfig(HumanBaseDataConfig):
    pass

class hot3dDataConfig(HumanBaseDataConfig):
    pass

class oakinkDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"

class seasmallDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"

class TacoDataConfig(HumanBaseDataConfig):
    video_backend = "torchvision_av"
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_rotation",
        "state.right_eef_position",
        "state.right_eef_rotation",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.right_eef_position",
        "action.right_eef_rotation",
    ]

class TASTE_robDataConfig(HumanBaseDataConfig):
    video_backend = "decord"
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    language_keys = ["annotation.language.action_text"]


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class EgoCentric10KDataConfig(HumanBaseDataConfig):
    video_backend = "decord"
    target_fps = 10
    video_keys = ["video.top_head"]
    future_video_keys = [
        "future_video.top_head"
    ]
    state_keys = [
        "state.qpos",
    ]
    action_keys = [
        "action.qpos",
    ]
    language_keys = ["annotation.language.action_text"]


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class DemoDataConfig:
    video_backend = "torchvision_av"
    video_keys = ["video.ego_view"]
    future_video_keys = [
        "future_video.ego_view"
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_width",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_width",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    future_observation_indices = [5]
    action_indices = list(range(0, 16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        future_video_modality = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "future_video": future_video_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys}
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

ROBOT_TYPE_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_eef": FourierGr1EEFDataConfig(),
    "fourier_gr1_arms_waist_twohistory": FourierGr1ArmsWaist_twohistoryDataConfig(),

    "agibot_gripper": AgibotWorldDataConfig(),
    "agibot_dex": AgibotDexDataConfig(),
    "galaxea": GalaxeaDataConfig(),
    "droid": DroidDataConfig(),
    "unitree": HumanoidEverydayDataConfig(),
    "intern_franka": FrankaDataConfig(),
    "intern_piper": InternDataConfig(),
    "intern_genie1": InternDataConfig(),
    "oxe": OxeDataConfig(),
    "robocoin_g1edu": RoboCoin_g1eduDataConfig(),
    "robocoin_leju": RoboCoin_lejuDataConfig(),
    "robocoin_r1lite": RoboCoin_r1liteDataConfig(),
    "ur": FrankaDataConfig(),
    "agilex":RobomindDataConfig(),
    "robomind_franka": FrankaDataConfig(),
    "robomind_franka_640": FrankaDataConfig(),
    "robomind_franka_dual": RobomindDataConfig(),

    "tienkung_gello": RobomindDataConfig(),
    "tienkung_xsens": RobomindDataConfig(),
    "r1pro": Challange2025DataConfig(),

    "egodex": EgodexDataConfig(),
    "hoi4d": Hoi4dDataConfig(),
    "holo_assist": HoloAssitDataConfig(),
    "hot3d": hot3dDataConfig(),
    "oakink": oakinkDataConfig(),
    "seasmall": seasmallDataConfig(),
    "taco": TacoDataConfig(),
    "taste_rob": TASTE_robDataConfig(),
    "egocentric_10k": EgoCentric10KDataConfig(),
    "vitra": VitraDataConfig(),
    "rh20t": RH20TDataConfig(),

    "demo_data": DemoDataConfig(),
}

