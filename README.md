# LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.03233-df2a2a.svg)](https://arxiv.org/abs/2602.12215)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://pku-epic.github.io/LDA/)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>

</details>

---

We introduce
**LDA-1B**, a robot foundation model that scales through universal
embodied data ingestion by jointly learning dynamics, policy,
and visual forecasting, assigning distinct roles to data of varying
quality.

![](assets/lda_teaser.png)

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🛠 Environment Setup](#-environment-setup)
- [🧩 Model Architecture](#-model-architecture)
- [💡 Training & Evaluation](#-training--evaluation)
- [🙏 Acknowledgements](#-acknowledgements)
- [✍️ Citation](#️-citation)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Unified Multi-Task Learning** | Single MMDiT backbone jointly predicts future visual features (`DINOv3` tokens) and 16-step action chunks |
| **Data Quality Hierarchy** | High-quality teleop → policy learning; Low-quality scripted → dynamics learning; No-annotation videos → visual forecasting |
| **Latent Dynamics Modeling** | Predicts future *latent visual features* instead of pixels → better generalization |
| **Cross-Embodiment** | Pre-trained on multi embodiments (Agibot, Unitree-G1, Human, etc.) |
---

## Latest Updates
- [2026-02-12] We publish LDA-1B, check our paper [here](https://arxiv.org/abs/2602.12215).


## 🛠 Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/jiangranlv/latent-dynamics-action.git LDA
cd LDA
```

### Step 2: Set Up Python Environment
Create and activate a conda environment with the required dependencies, for example:
```bash
# Create a conda environment
conda create -n LDA python=3.10
conda activate LDA

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2 with a version compatible with your PyTorch and CUDA versions
pip install flash-attn --no-build-isolation

# Install LDA
pip install -e .
```

### Step 3: Download Pretrained Model Weights

Follow the instruction in [Qwen3-VL](https://github.com/QwenLM/Qwen3) and [DINOv3](https://github.com/facebookresearch/dinov3) to download the pretrained VLM and vision encoder.

or you could directly download the pretrained from the following link:
   
   - `Qwen3-VL-4B`: [link](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)🤗
   
   - `DINO-ViT-S`: [link](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)🤗

## 🧩 Model Architecture

<div align="center">
<img src="./assets/LDA_pipeline.png" width="700" alt="LDA Architecture">
</div>

LDA jointly denoises action chunks and future visual latent under multiple co-training objectives. Conditioned on VLM tokens, diffusion
timesteps, and task embeddings, the model adopts a multimodal diffusion transformer architecture.

Core components:

- **Language and Vision Encoder**: Qwen3-VL (4B) → extracts semantics information

- **Latent Visual Representation**: DINOv3-ViT-S → extracts spatial features (frozen during training)

- **MM-DiT Backbone**: A 16-layer multi-modal diffusion transformer (`hidden_dim=1536`, `num_heads=32`).

Below is a description of the MM-DiT forward pass.

| Stage | Operation | Details |
|-------|-----------|---------|
| **1. Input Tokenization** | • **Image tokens**: DINOv3 patch embeddings (`[B, N_img, D]`)<br>• **Action tokens**: Linear projection of action chunks (`[B, N_act, D]`)<br>• **VLM tokens**: Qwen3-VL instruction embeddings (`[B, N_vlm, D]`) | All tokens share hidden dimension `D=1536` |
| **2. Self-Attention (Image + Action)** | • Image and action tokens compute **separate Q/K/V projections**<br>• Tokens are **concatenated** <br>• **Shared self-attention** over the combined sequence | Enables joint reasoning between visual observations and actions |
| **3. Cross-Attention (VLM → Image/Action)** | • VLM tokens serve as **queries**<br>• Image/action tokens serve as **keys&values**<br>• Two parallel cross-attention streams:<br>  &nbsp;&nbsp;– VLM → Image (for spatial grounding)<br>  &nbsp;&nbsp;– VLM → Action (for task conditioning) | The semantic information extracted by the VLM is incorporated into the generation process of action tokens and latent image tokens. |
| **4. AdaLN-Zero Conditioning** | Per-layer modulation of attention + MLP outputs via:<br>• **Diffusion timestep** `t` <br>• **Task embedding** (4-way categorical: *Policy* / *Forward Dynamics* / *Inverse Dynamics* / *Visual Forecasting*) | Dynamically adjusts model's behavior based on diffusion schedule and task objective |
| **5. Output Heads** | • **Latent dynamics head**: Predicts future DINOv3 tokens <br>• **Action head**: Predicts denoised 16-step action chunks | All four tasks are trained **jointly** within a single unified framework. |

## 💡 Training & Evaluation

### 🔥 Train LDA on RoboCasa-GR1 tabletop dataset

We provide training and evaluation scripts for the RoboCasa-GR1 dataset. Follow the steps described in [Robocasa_tabletop](examples/Robocasa_tabletop) to reproduce our results.

We also provide a [demo dataset](playground/demo_data) for quick debugging and validation.

### 🧪 Evaluate 

In addition to closed-loop evaluation in simulation (interactive execution with environment feedback), we also provide an open-loop evaluation interface for offline assessment. Open-loop evaluation quantitatively measures model performance by comparing predicted action sequences against ground-truth demonstrations from the dataset, without environment interaction.

```bash

bash LDA/scripts/eval_scripts/eval_lerobot_datasets_LDA.sh

```

## TODO

The following features are planned for future implementation:

- [ ] Pre-trained model checkpoints.
- [ ] Pre-training data.
- [ ] Data preprocess scripts.


##  🙏 Acknowledgements

Our code is built upon [starVLA](https://github.com/starVLA/starVLA) and [mmdit](https://gitlab.com/lucidrains/mmdit). These code serve as an essential foundation for our implementation, and we deeply appreciate the time, effort, and expertise they shared with the community.  

## ✍️ Citation


If you find our work useful, please cite us:


```
@article{lyu2026lda,
  title={LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion},
  author={Lyu, Jiangran and Liu, Kai and Zhang, Xuheng and Liao, Haoran and Feng, Yusen and Zhu, Wenxuan and Shen, Tingrui and Chen, Jiayi and Zhang, Jiazhao and Dong, Yifei and others},
  journal={arXiv preprint arXiv:2602.12215},
  year={2026}
}

```

## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

<!-- *Chart updates automatically. Click to interact with the full timeline.* -->
