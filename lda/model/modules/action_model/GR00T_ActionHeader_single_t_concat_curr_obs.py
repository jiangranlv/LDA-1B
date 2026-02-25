# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with lda, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field
import math
import os
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor
from einops import rearrange
import time
from typing import List

from lda.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from lda.model.modules.action_model.flow_matching_head.cdit import CDiT as DiT
from lda.model.modules.dinov3_vit import DINOv3ViTModel

TRAINING_TASKS = ["policy", "forward_dynamics", "inverse_dynamics", "video_gen"]

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple
def print_gpu_memory_usage(prefix=""):
    if torch.cuda.is_available():
        print(f"{prefix} GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
              f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

def get_dir_ckpt():
    """
    Alternative to gx_utils.file_manager.get_dir_ckpt().
    Returns the root directory of the project.
    """
    current_file = os.path.abspath(__file__)
    # Navigate to project root: action_head -> model -> gr00t -> World-Action-Model
    # From: /path/to/World-Action-Model/gr00t/model/action_head/flow_matching_action_head.py
    # To:   /path/to/World-Action-Model/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    return project_root

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        # self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        # self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))
        self.init_params()

    def init_params(self):
        # 对每个 category 独立初始化，模仿 nn.Linear
        for i in range(self.num_categories):
            tmp_linear = nn.Linear(self.W.shape[1], self.W.shape[2])  # in_dim -> hidden_dim
            self.W.data[i] = tmp_linear.weight.t().clone()  # 因为 Linear 是 (out, in)，我们存 (in, out)
            self.b.data[i] = tmp_linear.bias.clone()

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x

class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        time_horizon: int = 8,
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
        out_embed_dim: int = 1024,
        orig_patch_shape: tuple[int, ...] = None,
        glob_len: int = 0,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = time_horizon, orig_patch_shape[0], orig_patch_shape[1]
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW
        self.glob_len = glob_len
        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(out_embed_dim, num_chans * pT * pH * pW)
        if self.glob_len > 0:
            self.proj_glob = nn.Linear(num_chans, embed_dim)
            self.unproj_glob = nn.Linear(out_embed_dim, num_chans)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class MultiViewVideoPatchifierWithTimestep(nn.Module):
    """
    MultiViewVideoPatchifier with timestep conditioning for flow matching.
    Similar to MultiEmbodimentActionEncoder, this class concatenates image features
    with timestep embeddings for flow matching denoising.
    """
    def __init__(
        self,
        num_views: int,
        time_horizon: int = 8,
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
        out_embed_dim: int = 1024,
        orig_patch_shape: tuple[int, ...] = None,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = time_horizon, orig_patch_shape[0], orig_patch_shape[1]
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        # Patch encoder for images
        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        
        # Timestep conditioning layers (similar to MultiEmbodimentActionEncoder)
        self.W1 = nn.Linear(embed_dim, embed_dim)  # (w -> w)
        self.W2 = nn.Linear(2 * embed_dim, embed_dim)  # (2w -> w)
        self.W3 = nn.Linear(embed_dim, embed_dim)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim)
        
        # Decoder for unpatchify
        self.patch_decoder = nn.Linear(out_embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs, timesteps):
        """
        imgs: shape (B, V, C, T, H, W)
        timesteps: shape (B,) -- flow matching timestep per batch item
        returns: shape (B, V*T*H*W, embed_dim)
        """
        return self.patchify_with_timestep(imgs, timesteps)

    def patchify_with_timestep(self, imgs, timesteps):
        """
        Patchify images and condition on timesteps for flow matching.
        """
        B = imgs.shape[0]
        
        # 1) Standard patchify to get image features
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        img_feats = self.patch_encoder(imgs)  # (B*V, embed_dim, T, H, W)
        img_feats = rearrange(img_feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        # img_feats: (B, V*T*H*W, embed_dim)
        
        # 2) Expand timesteps across all patches
        num_patches = img_feats.shape[1]  # V*T*H*W
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B, num_patches)
            timesteps = timesteps.unsqueeze(1).expand(-1, num_patches)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across patches."
            )
        
        # 3) Get timestep embeddings
        tau_emb = self.pos_encoding(timesteps).to(dtype=img_feats.dtype)
        
        # 4) Apply timestep conditioning (similar to action encoder)
        # First pass through W1
        x = self.W1(img_feats)
        
        # Concat with timestep embedding and apply W2
        x = torch.cat([x, tau_emb], dim=-1)
        x = swish(self.W2(x))
        
        # Final pass through W3
        x = self.W3(x)
        
        return x

    def patchify(self, imgs):
        """Legacy method for backward compatibility"""
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    state_dim: int = field(default=None, metadata={"help": "State dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=1, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    vision_encoder_type: str = field(
        default=None, metadata={"help": "Use which vision encoder to encoder img"}
        ) # add vision encoder, choices = {"dinov3, vjepa"}
    vision_encoder_size: str = field(
        default="s", metadata={"help": "Use which size of vision encoder to encoder img"}
        )
    vision_encoder_path: str = field(
        default=None, metadata={"help": "Path to vision encoder."}
    )
    obs_horizon: int = field(
        default=8, metadata={"help": "Time horizon for future observation"}
    )
    num_views: int = field(
        default=1, metadata={"help": "number of img views"}
    )
    patch_shape: tuple[int] = field(
        default=(1, 1, 1), metadata={"help": "Resize img size"}
    )
    obs_loss_weight: float = field(
        default=1, metadata={"help": "Weight for observation loss"}
    )
    training_task_weights: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1],
        metadata={"help": "Weight for 4 training tasks"}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
    "DiT-XL": {"input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32},
}

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size # 是不要和 Q对齐？
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps
        self.vision_encoder_type = config.vision_encoder_type
        self.vision_encoder_size = config.vision_encoder_size
        self.obs_loss_weight = config.obs_loss_weight
        self.num_views = config.num_views
        self.training_task_weights = config.training_task_weights

        self.cross_attention_dim = config.diffusion_model_cfg['cross_attention_dim']

        self.inner_dim = action_model_cfg["num_attention_heads"] * action_model_cfg["attention_head_dim"]
        try:
            self.multi_embodiment = config.max_num_embodiments > 1
        except:
            config.max_num_embodiments = 1
        if config.max_num_embodiments > 1:
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,  
                output_dim=self.input_embedding_dim,
                ) if config.state_dim else None
            self.action_encoder = MultiEmbodimentActionEncoder(
                action_dim=config.action_dim,
                hidden_size=self.input_embedding_dim,
                num_embodiments=config.max_num_embodiments,
            )
            self.action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
        else:
            print("###########################################")
            print(f"Single Embodiment, using torch MLP")
            print("###########################################")
            self.state_encoder = MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            ) if config.state_dim else None

            self.action_encoder = ActionEncoder(
                action_dim=config.action_dim,
                hidden_size=self.input_embedding_dim,
            )
            self.action_decoder = MLP(
                input_dim=self.model.config.output_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
        # vision encoder
        assert self.vision_encoder_type is not None, "Vision encoder type is not set"
        if self.vision_encoder_type == "dinov3":
            pretrained_model_name = os.path.join(config.vision_encoder_path, f'dinov3-vit{self.vision_encoder_size}16-pretrain-lvd1689m')
            self.transform = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.vision_encoder = DINOv3ViTModel.from_pretrained(pretrained_model_name).eval()
            self.obs_horizon = config.obs_horizon
            self.cls_token = 1
            register_tokens = self.vision_encoder.config.num_register_tokens
            self.glob_len = self.cls_token + register_tokens
        elif self.vision_encoder_type == "vjepa2":
            pretrained_model_name = os.path.join(config.vision_encoder_path, f'vjepa2-vit{self.vision_encoder_size}-fpc64-256')
            self.transform= AutoVideoProcessor.from_pretrained(pretrained_model_name)
            self.vision_encoder = AutoModel.from_pretrained(pretrained_model_name).eval()
            self.tubelet_size = self.vision_encoder.config.tubelet_size # vjepa2 will patch time horzion as well
            self.obs_horizon = math.ceil(config.obs_horizon / self.tubelet_size)
            self.glob_len = 0
        self.img_size = self.vision_encoder.config.image_size
        num_chans = self.vision_encoder.config.hidden_size
        s = self.vision_encoder.config.image_size // self.vision_encoder.config.patch_size
        self.orig_patch_shape = (s, s)
        self.obs_len = self.orig_patch_shape[0] * self.orig_patch_shape[1] * self.obs_horizon * self.num_views

        # align concat obs embedding dim with other tokens
        self.obs_merger = nn.Linear(num_chans * 2, self.input_embedding_dim)

        self.obs_projector = nn.Linear(self.hidden_size, num_chans)
        
        # register tokens, TODO: maybe need to be cancelled 
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # learnable tokens 
        self.next_obs_learnable_tokens = nn.Parameter(0.02 * torch.randn(num_chans))

        self.action_learnable_tokens = nn.Embedding(self.action_horizon, self.input_embedding_dim)
        nn.init.normal_(self.action_learnable_tokens.weight, mean=0.0, std=0.02)
        
        # task related embeddings, will be added to timestep embedding
        self.policy_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))
        self.fd_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))
        self.vg_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))
        self.id_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    def encode_future_img(self, next_obs, microbatch_size=72):
        if self.vision_encoder_type == 'vjepa2':
            next_obs = rearrange(next_obs, "b v t c h w -> (b v) t c h w")
            with torch.no_grad():
                next_obs = self.vision_encoder.get_vision_features(next_obs)
            next_obs = rearrange(next_obs, "b (t h w) d -> b t h w d", h=self.orig_patch_shape[0], w=self.orig_patch_shape[1])
        elif self.vision_encoder_type == 'dinov3':
            next_obs = rearrange(next_obs, "b v t c h w -> (b v t) c h w")
            transformed_imgs = []
            for i in range(0, next_obs.shape[0], microbatch_size):
                batch_next_obs = next_obs[i : i + microbatch_size]
                with torch.no_grad():
                    output = self.vision_encoder(batch_next_obs)
                batch_next_obs = output.last_hidden_state
                transformed_imgs.append(batch_next_obs)
            next_obs = torch.cat(transformed_imgs, dim=0) # (B, N, D)

        return next_obs
    def transform_obs(self, obs, B, V, T):
            if len(obs.shape) == 6:
                obs = rearrange(obs, "b v t c h w -> (b v t) c h w")
            if self.vision_encoder_type == "vjepa2":
                obs = self.transform(obs)["pixel_values_videos"][0]
                obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=V, t=T)
            else:
                obs = torch.stack(self.transform(obs)["pixel_values"], dim=0)
                obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=V, t=T)
            return obs

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        # action_mask: torch.Tensor,
        state: torch.Tensor = None,
        future_imgs: torch.Tensor = None,
        curr_imgs: torch.Tensor = None,
        embodiment_id: torch.Tensor = None,
        assigned_tasks: List[str] = None,  
        encoder_attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            vl_embs: (B, seq_len, D)
            actions: (B, T, action_dim)
            action_mask: (B, T, action_dim)
            state: (B, state_dim) [optional]
            future_imgs: (B, V*T, C, H, W)
            curr_imgs: (B, V*T, C, H, W)
            embodiment_id: (B,)
            assigned_tasks: List[str] of length B, e.g. ["policy", "video_gen", ...]
        """
        device = vl_embs.device
        B = vl_embs.shape[0]

        if assigned_tasks is None:
            raise ValueError("assigned_tasks must be provided for strict task-balanced training.")

        # === 1. 预处理 curr_obs / next_obs ===
        curr_obs = rearrange(curr_imgs, "b (v t) c h w -> b v t c h w", v=self.num_views)
        B_, V, T = curr_obs.shape[:3]
        assert B_ == B

        curr_obs = self.transform_obs(curr_obs, B, V, T)
        curr_obs = self.encode_future_img(curr_obs)
        if self.vision_encoder_type == "vjepa2":
            curr_obs = rearrange(curr_obs, "(b v) t h w c -> b (v t h w) c", b=B, v=V)
        else:
            curr_obs = rearrange(curr_obs, "(b v t) n c -> b (v t n) c", b=B, v=V)
        # === 2. 处理 next_obs（real or learnable）===
        next_obs = rearrange(future_imgs, "b (v t) c h w -> b v t c h w", v=self.num_views)
        next_obs = self.transform_obs(next_obs, B, V, T)
        next_obs = self.encode_future_img(next_obs)
        if self.vision_encoder_type == "vjepa2":
            next_obs = rearrange(next_obs, "(b v) t h w c -> b (v t h w) c", b=B, v=V)
        else:
            next_obs = rearrange(next_obs, "(b v t) n c -> b (v t n) c", b=B, v=V)

        num_obs_tokens = next_obs.shape[1]

        # === 3. State embedding ===
        if self.multi_embodiment:
            state_features = self.state_encoder(state, embodiment_id) if self.state_dim is not None else None
        else:
            state_features = self.state_encoder(state) if self.state_dim is not None else None

        # === 4. 准备 per-sample 输入 ===
        # 为每个样本构造：action_features, noisy_next_obs, velocity
        policy_indices = []
        forward_dynamics_indices = []
        inverse_dynamics_indices = []
        video_gen_indices = []

        for i in range(B):
            task = assigned_tasks[i] 
            if task == "policy":
                policy_indices.append(i)
            elif task == "forward_dynamics":
                forward_dynamics_indices.append(i)
            elif task == "inverse_dynamics":
                inverse_dynamics_indices.append(i)
            elif task == "video_gen":
                video_gen_indices.append(i)

        # set task embedding
        policy_embedding = self.policy_embedding.unsqueeze(0).expand(len(policy_indices), -1)
        fd_embedding = self.fd_embedding.unsqueeze(0).expand(len(forward_dynamics_indices), -1)
        vg_embedding = self.vg_embedding.unsqueeze(0).expand(len(video_gen_indices), -1)
        id_embedding = self.id_embedding.unsqueeze(0).expand(len(inverse_dynamics_indices), -1)

        pred_action_task_indices = policy_indices + inverse_dynamics_indices
        pred_next_obs_task_indices = forward_dynamics_indices + video_gen_indices

        # policy and inverse dynamics 
        policy_action = actions[policy_indices]
        inverse_action = actions[inverse_dynamics_indices]
        
        to_noise_action = torch.cat((policy_action, inverse_action), dim=0)
        act_t_sample = self.sample_time(to_noise_action.shape[0], device=device, dtype=vl_embs.dtype).squeeze()
        action_noise = torch.randn_like(to_noise_action)
        act_t_sample = act_t_sample[:, None, None]
         # noisy action
        noisy_action = (1 - act_t_sample) * action_noise + act_t_sample * to_noise_action
        action_velocity = to_noise_action - action_noise

        act_t_discretized = (act_t_sample[:, 0, 0] * self.num_timestep_buckets).long()
        if self.multi_embodiment:
            noisy_act_feat = self.action_encoder(noisy_action, act_t_discretized, embodiment_id[pred_action_task_indices])
        else:
            noisy_act_feat = self.action_encoder(noisy_action, act_t_discretized)  # (1, T, D)
        # policy: use learnable_next_obs_tokens to replace next obs
        policy_obs_feat = self.next_obs_learnable_tokens.unsqueeze(0).unsqueeze(0).expand(len(policy_indices), num_obs_tokens, -1)
        # inverse_dynamics: use gt next obs tokens
        inv_obs_feat = next_obs[inverse_dynamics_indices]

        # forward dynamics and video gen
        forward_obs = next_obs[forward_dynamics_indices]
        video_gen_obs = next_obs[video_gen_indices]
        to_noise_next_obs = torch.cat((forward_obs, video_gen_obs), dim=0)
        obs_t_sample = self.sample_time(to_noise_next_obs.shape[0], device=device, dtype=vl_embs.dtype).squeeze()
        obs_t = obs_t_sample[:, None, None]
        obs_noise = torch.randn_like(to_noise_next_obs)
        noisy_obs = (1 - obs_t) * obs_noise + obs_t * to_noise_next_obs
        obs_velocity = to_noise_next_obs - obs_noise
        # if self.vision_encoder_type == "vjepa2":
        #     noisy_obs_shape = noisy_obs.shape[:-2]
        #     obs_t_discretized = (obs_t[:, 0, 0, 0, 0, 0] * self.num_timestep_buckets).long()
        # else:
        obs_t_discretized = (obs_t[:, 0, 0] * self.num_timestep_buckets).long()

        # forward dynamics: use gt action
        t_clean = torch.ones(len(forward_dynamics_indices), device=device, dtype=vl_embs.dtype)
        t_discretized_clean = (t_clean * self.num_timestep_buckets).long()
        if self.multi_embodiment:
            forward_act_feat = self.action_encoder(
                actions[forward_dynamics_indices], t_discretized_clean, embodiment_id[forward_dynamics_indices]
            )
        else:
            forward_act_feat = self.action_encoder(
                actions[forward_dynamics_indices], t_discretized_clean
            )
        # video gen: use learnable action tokens
        video_gen_act_feat = self.action_learnable_tokens.weight.unsqueeze(0).expand(len(video_gen_indices), -1, -1)

        # === 5. 拼接所有样本 ===
        action_features = torch.cat((noisy_act_feat, forward_act_feat, video_gen_act_feat), dim=0)  # (B, T_a, D)
        noisy_next_obs = torch.cat((policy_obs_feat, inv_obs_feat, noisy_obs), dim=0)    # (B, N_obs, D)
        diffusion_t = torch.cat((act_t_discretized, obs_t_discretized), dim=0)          # (B,)    
        task_embedding = torch.cat((policy_embedding, id_embedding, fd_embedding, vg_embedding), dim=0)
        # === 6. 构建完整输入序列 ===
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        # concat curr_obs_tokens with future_tokens channelwise
        obs_tokens = self.obs_merger(torch.cat([curr_obs, noisy_next_obs], dim=-1))
        if state_features is not None:
            sa_embs = torch.cat([state_features, future_tokens, action_features, obs_tokens], dim=1)
        else:
            sa_embs = torch.cat([future_tokens, action_features, obs_tokens], dim=1)

        if self.config.add_pos_embed:
            total_len = sa_embs.shape[1]
            pos_ids = torch.arange(total_len, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = sa_embs + pos_embs
        # === 7. 模型前向 ===
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=diffusion_t,
            return_all_hidden_states=False,
            task_embedding=task_embedding,
            encoder_attention_mask=encoder_attention_mask,
        )

        # === 8. 解码输出 ===
        if self.multi_embodiment:
            pred_actions = self.action_decoder(model_output, embodiment_id)
        else:
            pred_actions = self.action_decoder(model_output)
        pred_actions = pred_actions[:, -(actions.shape[1] + self.obs_len + self.glob_len) : -(self.obs_len + self.glob_len)]
        
        # === 9. 计算 per-sample loss ===
        total_loss = 0.0

        pred_actions = pred_actions[:len(pred_action_task_indices)]
        # mask = action_mask[pred_action_task_indices]
        # act_loss = F.mse_loss(pred_actions, action_velocity, reduction="none") * mask
        act_loss = F.mse_loss(pred_actions, action_velocity, reduction="mean")
        # act_loss = act_loss.sum() / (mask.sum() + 1e-8)
        total_loss += act_loss

        pred_next_obs = self.obs_projector(model_output)
        pred_next_obs = pred_next_obs[-len(pred_next_obs_task_indices):, -(self.obs_len + self.glob_len):]
        obs_loss = F.mse_loss(pred_next_obs, obs_velocity)
        total_loss += obs_loss
        return BatchFeature(data={
            "loss": total_loss,
            "action_loss": act_loss.detach(),
            "dynamics_loss": obs_loss.detach(),
        })

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        curr_imgs: torch.Tensor = None,
        embodiment_id: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Denoising diffusion sampling for action prediction (policy task only).
        """
        device = vl_embs.device
        B = vl_embs.shape[0]

        # === 1. Encode current observation (same as in forward) ===
        curr_obs = rearrange(curr_imgs, "b (v t) c h w -> b v t c h w", v=self.num_views)
        B_, V, T = curr_obs.shape[:3]
        assert B_ == B

        curr_obs = self.transform_obs(curr_obs, B, V, T)
        curr_obs = self.encode_future_img(curr_obs)
        if self.vision_encoder_type == "vjepa2":
            curr_obs = rearrange(curr_obs, "(b v) t h w c -> b (v t h w) c", b=B, v=V)
        else:
            curr_obs = rearrange(curr_obs, "(b v t) n c -> b (v t n) c", b=B, v=V)

        num_obs_tokens = curr_obs.shape[1]
        # === 2. Initialize noisy action (sample from N(0, I)) ===
        actions = torch.randn(
            size=(B, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        # === 3. State features (if provided) ===
        if self.multi_embodiment:
            state_features = self.state_encoder(state, embodiment_id) if self.state_dim is not None else None
        else:
            state_features = self.state_encoder(state) if self.state_dim is not None else None

        # === 4. Task embedding: only "policy" during inference ===
        task_embedding = self.policy_embedding.unsqueeze(0).expand(B, -1)  # (B, D)

        # === 5. Denoising loop ===
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cont = step / float(num_steps)  # [0, 1)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((B,), t_discretized, device=device, dtype=torch.long)

            # === 5.1 Encode noisy actions ===
            if self.multi_embodiment:
                action_features = self.action_encoder(actions, timesteps, embodiment_id)
            else:
                action_features = self.action_encoder(actions, timesteps)  # (B, T_a, D)

            # === 5.2 Noisy next obs: policy uses learnable tokens (same as forward) ===
            noisy_next_obs = self.next_obs_learnable_tokens.unsqueeze(0).expand(B, num_obs_tokens, -1)  # (B, N_obs, D)

            # === 5.3 Future tokens ===
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)  # (B, T_f, D)

            obs_tokens = self.obs_merger(torch.cat([curr_obs, noisy_next_obs], dim=-1))
            # === 5.4 Assemble full input sequence (same order as forward) ===
            if state_features is not None:
                sa_embs = torch.cat([state_features, future_tokens, action_features, obs_tokens], dim=1)
            else:
                sa_embs = torch.cat([future_tokens, action_features, obs_tokens], dim=1)

            # === 5.5 Add positional embedding (if enabled) ===
            if self.config.add_pos_embed:
                total_len = sa_embs.shape[1]
                pos_ids = torch.arange(total_len, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                sa_embs = sa_embs + pos_embs

            # === 5.6 Forward through backbone ===
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
                return_all_hidden_states=False,
                task_embedding=task_embedding,  
                encoder_attention_mask=encoder_attention_mask,
            )

            # === 5.7 Extract and decode action velocity ===
            if self.multi_embodiment:
                pred_actions_tokens = self.action_decoder(model_output, embodiment_id)
            else:
                pred_actions_tokens = self.action_decoder(model_output)
            pred_velocity = pred_actions_tokens[:, -(actions.shape[1] + self.obs_len + self.glob_len) : -(self.obs_len + self.glob_len)]

            # === 5.8 Euler integration step ===
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass