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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = 'franka'
    """
    The Franka Emika Panda robot.
    """

    FRANKA_DUAL = 'franka_dual'

    EGOVLA = "egovla"
    """
    The EgoVLA dataset
    """

    GALBOT = "galbot"
    """
    The Galbot dataset
    """

    EGOCENTRIC_10K = "egocentric_10k"
    """
    The EgoCentric-10K dataset
    """

    AGIBOT_GRIPPER = "agibot_gripper"

    AGIBOT_DEX = "agibot_dex"

    Galaxea = "galaxea"

    Droid = "droid"

    Unitree = 'unitree'

    PIPER = "piper"

    R1PRO = "r1pro"

    GENIE1 = "genie1"

    OXE = "oxe"

    RoboCoin_g1edu = "robocoin_g1edu"

    RoboCoin_leju = "robocoin_leju"

    RoboCoin_r1lite = "robocoin_r1lite"

    AGILEX = "agilex"

    TIENKUNG_GELLO = "tienkung_gello"

    TIENKUNG_XSENS = "tienkung_xsens"

    UR = "ur"

    Vitra = "vitra"

    Egodex = "egodex"

    Hoi4d = "hoi4d"

    HoloAssit = "holo_assist"

    hot3d = "hot3d"

    oakink = "oakink"

    seasmall = "seasmall"

    Taco = "taco"

    TASTE_Rob = "taste_rob"



TASK_MAPPING = {
    # define trainable tasks for different datasets
    EmbodimentTag.EGOVLA.value: ['policy', 'forward_dynamics', 'inverse_dynamics', 'video_gen'],
    EmbodimentTag.GR1.value: ['policy', 'forward_dynamics', 'inverse_dynamics', 'video_gen'],
    EmbodimentTag.GALBOT.value: ['policy', 'forward_dynamics', 'inverse_dynamics', 'video_gen'],
    EmbodimentTag.EGOCENTRIC_10K.value: ['video_gen'],
    EmbodimentTag.TASTE_Rob.value: ['video_gen']
}

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.AGIBOT_GRIPPER.value: 0,

    EmbodimentTag.Galaxea.value: 1,
    EmbodimentTag.Droid.value: 2,
    EmbodimentTag.Unitree.value: 3,
    EmbodimentTag.FRANKA.value: 4,

    EmbodimentTag.OXE.value: 5,
    EmbodimentTag.RoboCoin_g1edu.value: 6,
    EmbodimentTag.RoboCoin_leju.value: 7,
    EmbodimentTag.RoboCoin_r1lite.value: 8,
    EmbodimentTag.UR.value: 9,
    EmbodimentTag.Vitra.value: 10,
    EmbodimentTag.Egodex.value: 11,
    EmbodimentTag.Hoi4d.value: 12,
    EmbodimentTag.HoloAssit.value: 13,
    EmbodimentTag.hot3d.value: 14,
    EmbodimentTag.oakink.value: 15,
    EmbodimentTag.seasmall.value: 16,
    EmbodimentTag.Taco.value: 17,
    EmbodimentTag.PIPER.value: 18,
    EmbodimentTag.R1PRO.value: 19,
    EmbodimentTag.GENIE1.value: 20,
    EmbodimentTag.AGIBOT_DEX.value: 21,
    EmbodimentTag.TASTE_Rob.value: 22,
    EmbodimentTag.AGILEX.value: 23,

    EmbodimentTag.GR1.value: 24, 
    EmbodimentTag.TIENKUNG_GELLO.value: 25,
    EmbodimentTag.TIENKUNG_XSENS.value: 26,
    EmbodimentTag.GALBOT.value: 27,
    EmbodimentTag.EGOCENTRIC_10K.value: 28,
    EmbodimentTag.EGOVLA.value: 29,
    EmbodimentTag.FRANKA_DUAL.value: 30,

    EmbodimentTag.NEW_EMBODIMENT.value: 31,
}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
    "fourier_gr1_eef": EmbodimentTag.GR1,
    "fourier_gr1_arms_waist_twohistory": EmbodimentTag.GR1,
    "egovla": EmbodimentTag.EGOVLA,
    "galbot": EmbodimentTag.GALBOT,
    "egocentric_10k": EmbodimentTag.EGOCENTRIC_10K,


    "agibot_gripper": EmbodimentTag.AGIBOT_GRIPPER,
    "agibot_dex": EmbodimentTag.AGIBOT_DEX,
    "galaxea": EmbodimentTag.Galaxea,
    "droid": EmbodimentTag.Droid,
    "unitree": EmbodimentTag.Unitree,
    "robomind_franka": EmbodimentTag.FRANKA,
    "robomind_franka_640": EmbodimentTag.FRANKA,
    "robomind_franka_dual": EmbodimentTag.FRANKA_DUAL,
    "r1pro": EmbodimentTag.R1PRO,
    "intern_piper": EmbodimentTag.PIPER,
    "intern_franka": EmbodimentTag.FRANKA,
    "intern_genie1": EmbodimentTag.GENIE1,
    "oxe": EmbodimentTag.OXE,
    "robocoin_g1edu": EmbodimentTag.RoboCoin_g1edu,
    "robocoin_leju": EmbodimentTag.RoboCoin_leju,
    "robocoin_r1lite": EmbodimentTag.RoboCoin_r1lite,
    "agilex": EmbodimentTag.AGILEX,
    "vitra": EmbodimentTag.Vitra,
    "egodex": EmbodimentTag.Egodex,
    "hoi4d": EmbodimentTag.Hoi4d,
    "holo_assist": EmbodimentTag.HoloAssit,
    "hot3d": EmbodimentTag.hot3d,
    "oakink": EmbodimentTag.oakink,
    "seasmall": EmbodimentTag.seasmall,
    "taco": EmbodimentTag.Taco,
    "taste_rob": EmbodimentTag.TASTE_Rob,
    "ur": EmbodimentTag.UR,
    "tienkung_gello": EmbodimentTag.TIENKUNG_GELLO,
    "tienkung_xsens": EmbodimentTag.TIENKUNG_XSENS,

    "unitree_g1": EmbodimentTag.Unitree,
    "galbot_sharpa": EmbodimentTag.GALBOT,

    "demo_data": EmbodimentTag.NEW_EMBODIMENT,
}
