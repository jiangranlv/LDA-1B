# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with LDA, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig, CLIPTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor
from einops import rearrange
import time

from LDA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from LDA.model.modules.action_model.flow_matching_head.uwm_cross_attention_dit import DiT
from LDA.model.modules.dinov3_vit import DINOv3ViTModel
from LDA.model.modules.action_model.UWM.transforms import VAEDownsample, VideoTransform
from LDA.model.modules.action_model.UWM.language import CLIPTextEncoder
from LDA.model.modules.action_model.UWM.vision import ViTImageEncoder, ResNetImageEncoder
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
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
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
        default='dinov3', metadata={"help": "Use which vision encoder to encoder img"}
        ) # add vision encoder, choices = {"dinov3, vjepa"}
    vision_encoder_size: str = field(
        default="s", metadata={"help": "Use which size of vision encoder to encoder img"}
        )
    vision_encoder_path: str = field(
        default=None, metadata={"help": "Path to vision encoder."}
    )
    use_img_denoise: bool = field(
        default=False, metadata={"help": "Whether to predict next img"}
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
        self.hidden_size = config.hidden_size # @JinhuiYE
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps
        self.vision_encoder_type = config.vision_encoder_type
        self.vision_encoder_size = config.vision_encoder_size
        self.num_views = config.num_views

        self.cross_attention_dim = config.diffusion_model_cfg['cross_attention_dim']

        self.inner_dim = action_model_cfg["num_attention_heads"] * action_model_cfg["attention_head_dim"]
        # self.img_size = config.img_shape

        # LDA only support single embodiment, if use multi embodiment, replace with multiMLP
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

        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.text_encoder = (
            CLIPTextEncoder(embed_dim=self.inner_dim) 
        )

        self.image_encoder = ViTImageEncoder(
            num_views=config.num_views,
            embed_dim=self.inner_dim
        )

        self.use_img_denoise = config.use_img_denoise
        # add vision encoder 
        if self.use_img_denoise:
            assert self.vision_encoder_type is not None, "Vision encoder type is not set"
            if self.vision_encoder_type == "dinov3":
                # pretrained_model_name = os.path.join(get_dir_ckpt(), f'pretrained/dinov3-vit{self.vision_encoder_size}16-pretrain-lvd1689m')
                pretrained_model_name = os.path.join(config.vision_encoder_path, f'dinov3-vit{self.vision_encoder_size}16-pretrain-lvd1689m')
                self.transform = AutoImageProcessor.from_pretrained(pretrained_model_name)
                self.vision_encoder = DINOv3ViTModel.from_pretrained(pretrained_model_name).eval()
                self.obs_horizon = config.obs_horizon
                self.cls_token = 1
                self.register_tokens = self.vision_encoder.config.num_register_tokens
                self.glob_len = self.cls_token + self.register_tokens
            elif self.vision_encoder_type == "vjepa2":
                pretrained_model_name = os.path.join(config.vision_encoder_path, f'vjepa2-vit{self.vision_encoder_size}-fpc64-256')
                self.transform= AutoVideoProcessor.from_pretrained(pretrained_model_name)
                self.vision_encoder = AutoModel.from_pretrained(pretrained_model_name).eval()
                self.tubelet_size = self.vision_encoder.config.tubelet_size # vjepa2 will patch time horzion as well
                self.obs_horizon = math.ceil(config.obs_horizon / self.tubelet_size)
                self.glob_len = 0
            elif self.vision_encoder_type == "vae":
                self.vision_encoder = VAEDownsample(config.vision_encoder_path)
                self.transform = VideoTransform(
                    resize_shape=(224, 224),
                    imagenet_norm=True,
                )
                self.glob_len = 0
                self.obs_horizon = config.obs_horizon
            patch_shape, self.latent_img_shape = self.vision_encoder.latent_img_shape(self.obs_horizon)
            num_chans = self.latent_img_shape[0]

            self.img_size = (224,224)
            self.orig_patch_shape = patch_shape
            # Ensure patch_shape is a tuple (OmegaConf may load it as a list)
            latent_patch_shape = [1, 4, 4]  # (pT, pH, pW)

            self.obs_patchifier = MultiViewVideoPatchifier(
                    num_views=config.num_views,
                    # input_shape=config.img_shape,
                    time_horizon=self.obs_horizon,
                    patch_shape=latent_patch_shape,
                    num_chans=num_chans,
                    embed_dim=self.input_embedding_dim,
                    out_embed_dim=self.hidden_size,
                    orig_patch_shape=self.orig_patch_shape,
                    glob_len=self.glob_len,
                )
            
            self.patch_len = self.orig_patch_shape[0] * self.orig_patch_shape[1]
            self.obs_len = self.obs_patchifier.num_patches + self.glob_len * config.num_views * self.obs_horizon

            
            # else:
            #     raise NotImplementedError(f"Unsupported Vision Encoder type: {self.vision_encoder_type}"
            # project curr obs to cross attention dim, for concat with vlm embs
            # NOTE:use which method to merge curr obs to the dit
            # if self.concat_curr_obs_with_vlm_embs: 
            #     self.curr_obs_encoder = nn.Linear(num_chans, self.cross_attention_dim) if self.cross_attention_dim != self.input_embedding_dim else nn.Identity()
            # else:
            #     self.curr_obs_encoder = nn.Linear(num_chans, self.inner_dim) if self.inner_dim != self.input_embedding_dim else nn.Identity() # concat curr obs with temb
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

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
        elif self.vision_encoder_type == "vae":
            next_obs = rearrange(next_obs, "b v t c h w -> (b v t) c h w")
            transformed_img = self.vision_encoder(next_obs)
            next_obs = transformed_img

        return next_obs
    def transform_obs(self, obs, B, V, T):
        if len(obs.shape) == 6:
            obs = rearrange(obs, "b v t c h w -> (b v t) c h w")
        if self.vision_encoder_type == "vjepa2":
            obs = self.transform(obs)["pixel_values_videos"][0]
            obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=V, t=T)
        elif self.vision_encoder_type == "dinov3":
            obs = torch.stack(self.transform(obs)["pixel_values"], dim=0)
            obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=V, t=T)
        elif self.vision_encoder_type == "vae":
            obs = rearrange(obs, "(b v t) c h w -> b (v t) h w c", b=B, v=V, t=T)
            obs = self.transform(obs)
            obs = rearrange(obs, "b c (v t) h w -> b v t c h w", b=B, v=V, t=T)
        return obs
    
    def patchify_obs_with_timestep(self, obs, timestep=None):
        """
        Helper method to patchify observations with optional timestep conditioning.
        If timestep is None, uses the legacy patchify method.
        """
        if timestep is not None:
            return self.obs_patchifier(obs, timestep)
        else:
            return self.obs_patchifier.patchify(obs)

    def forward(self, actions: torch.Tensor, action_mask: torch.Tensor, state: torch.Tensor = None, 
                future_imgs: torch.Tensor = None, curr_imgs: torch.Tensor = None, embodiment_id: torch.Tensor = None, 
                langs: list[str] = None) -> torch.Tensor:
        """
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device

        # Encode language instructions
        if langs is not None:
            text_inputs = self.text_tokenizer(langs, padding=True, return_tensors='pt', truncation=True).to(device)
            text_embs = self.text_encoder(text_inputs.input_ids, text_inputs.attention_mask)  # (B, L, D)

        # embed state
        state_features = self.state_encoder(state).squeeze(1) if state is not None else None

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        action_t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        action_t = action_t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - action_t) * noise + action_t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        action_t_discretized = (action_t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, action_t_discretized)

        # Embed noised future obs, need to be refined, current only take single img 
        obs_t_discretized = None
        curr_obs = None
        if self.use_img_denoise:
            curr_obs = rearrange(curr_imgs, "b (v t) c h w -> b v t c h w", v=self.num_views)    #(B, V, T, C, H, W)
            next_obs = rearrange(future_imgs, "b (v t) c h w -> b v t c h w", v=self.num_views) # (B, V, T, C, H, W)
            B, V, T= next_obs.shape[:3]
            curr_obs = self.transform_obs(curr_obs, B, V, T)
            next_obs = self.transform_obs(next_obs, B, V, T)
            curr_obs = self.image_encoder(curr_obs)
            curr_obs = torch.cat([curr_obs, state_features], dim=-1) if state_features is not None else curr_obs
            curr_obs = torch.cat([curr_obs, text_embs], dim=-1) if langs is not None else curr_obs

            next_obs = self.encode_future_img(next_obs) # (B, D, H, W)
            if self.vision_encoder_type == "vjepa2":
                next_obs = rearrange(next_obs, "(b v) t h w c -> b v c t h w", b=B, v=V)
            elif self.vision_encoder_type == "dinov3":
                next_obs = rearrange(next_obs, "(b v t) n c -> b v c t n", b=B, v=V)
            elif self.vision_encoder_type == "vae":
                next_obs = rearrange(next_obs, "(b v t) c h w -> b v c t h w", b=B, v=V)
            obs_noise = torch.randn(next_obs.shape, device=next_obs.device, dtype=next_obs.dtype)
            obs_t = self.sample_time(next_obs.shape[0], device=next_obs.device, dtype=next_obs.dtype)
            if self.vision_encoder_type == "dinov3":
                obs_t = obs_t[:, None, None, None, None]
            else:
                obs_t = obs_t[:, None, None, None, None, None]

            noisy_next_obs = (1 - obs_t) * obs_noise + obs_t * next_obs
            obs_velocity = next_obs - obs_noise

            if self.vision_encoder_type == "dinov3":
                obs_shape = noisy_next_obs.shape[:-1]
            else:
                obs_shape = noisy_next_obs.shape[:-2]
            # curr_obs_shape = curr_obs.shape[:-2]
            # Use timestep conditioning for flow matching
            # obs_t_for_patchifier = (obs_t[:, 0, 0, 0, 0, 0] * self.num_timestep_buckets).long() # Extract scalar timestep
            # curr_obs_t = (torch.ones(curr_obs.shape[0], device=curr_obs.device, dtype=curr_obs.dtype) * self.num_timestep_buckets).long()
            # curr_obs = self.patchify_obs_with_timestep(curr_obs.reshape(*obs_shape, *self.orig_patch_shape), curr_obs_t)
            # noisy_next_obs = self.patchify_obs_with_timestep(noisy_next_obs.reshape(*obs_shape, *self.orig_patch_shape), obs_t_for_patchifier)
            def to_tokens(obs):
                if self.glob_len > 0:
                    if self.vision_encoder_type=="vjepa2":
                        obs = obs.reshape(*obs_shape, -1)
                    local_tokens = self.obs_patchifier(obs[..., -self.patch_len:].reshape(*obs_shape, *self.orig_patch_shape))
                    glob_tokens = self.obs_patchifier.proj_glob(obs[..., :-self.patch_len].permute(0, 1, 3, 4, 2).reshape(len(local_tokens), -1, obs.shape[2]))
                    return torch.cat([glob_tokens, local_tokens], dim=-2)
                else:
                    return self.obs_patchifier(obs.reshape(*obs_shape, *self.orig_patch_shape))

            noisy_next_obs = to_tokens(noisy_next_obs) # B, N, D

            if self.vision_encoder_type == "dinov3":
                obs_t_discretized = (obs_t[:, 0, 0, 0, 0] * self.num_timestep_buckets).long()
            else:
                obs_t_discretized = (obs_t[:, 0, 0, 0, 0, 0] * self.num_timestep_buckets).long()

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            if self.use_img_denoise:
                total_len = action_features.shape[1] + noisy_next_obs.shape[1] + future_tokens.shape[1]
            else:
                total_len = action_features.shape[1]
            pos_ids = torch.arange(total_len, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0) 

        # Join VLM features with state and action embedding along sequence dimension.
        if self.use_img_denoise:
            sa_embs = torch.cat((future_tokens, action_features, noisy_next_obs), dim=1)
            sa_embs = sa_embs + pos_embs if self.config.add_pos_embed else sa_embs
        else:
            action_features = action_features + pos_embs if self.config.add_pos_embed else action_features
            sa_embs = torch.cat((future_tokens, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=curr_obs,
            timestep=action_t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
            obs_timestep=obs_t_discretized,
        )
        if self.use_img_denoise:
            next_obs_noise_pred = model_output[:, -self.obs_len :]
            pred_actions = model_output[:, -(actions.shape[1] + self.obs_len) : -self.obs_len]
            if self.glob_len == 0:
                next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
            else:
                glob_noise_pred = next_obs_noise_pred[:, :-self.obs_patchifier.num_patches]
                glob_noise_pred = self.obs_patchifier.unproj_glob(glob_noise_pred)
                glob_noise_pred = glob_noise_pred.reshape(len(glob_noise_pred), obs_shape[1], obs_shape[3], -1, glob_noise_pred.shape[-1]).permute(0, 1, 4, 2, 3)
                patch_noise_pred = next_obs_noise_pred[:, -self.obs_patchifier.num_patches:]
                patch_noise_pred = self.obs_patchifier.unpatchify(patch_noise_pred)
                patch_noise_pred = patch_noise_pred.reshape(*obs_shape, -1)
                next_obs_noise_pred = torch.cat([glob_noise_pred, patch_noise_pred], dim=-1)
        else:
            pred_actions = model_output[:, -actions.shape[1] :]
        pred_actions = self.action_decoder(pred_actions)
        # Slice out only the action portion of pred and target.
        # action_loss = ((pred_actions - velocity) ** 2).mean()
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        action_loss = action_loss.sum() / action_mask.sum() 
        if self.use_img_denoise:
            obs_loss = ((next_obs_noise_pred - obs_velocity) ** 2).mean()   
            loss = action_loss +  obs_loss
        else:
            loss = action_loss
        output_dict = {
            "loss": loss,
            "action_loss": action_loss.detach(),
            "dynamics_loss": obs_loss.detach() if self.use_img_denoise else 0,
        }
 
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def predict_action(
        self,
        state: torch.Tensor,
        curr_imgs: torch.Tensor,          # (B, V*T, C, H, W)
        embodiment_id: torch.Tensor = None,
        langs: list[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Inference-time action prediction via diffusion sampling.
        Inputs are raw (not pre-encoded), matching the format used in `forward`.
        """
        batch_size = curr_imgs.shape[0]
        device = curr_imgs.device

        # === 1. Encode language ===
        text_embs = None
        if langs is not None:
            text_inputs = self.text_tokenizer(
                langs, padding=True, return_tensors='pt', truncation=True
            ).to(device)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                text_embs = self.text_encoder(text_inputs.input_ids, text_inputs.attention_mask)
            # text_embs: (B, L, D)
        dtype = text_embs.dtype
        # === 2. Encode current observation (for cross-attention) ===
        curr_obs = rearrange(curr_imgs, "b (v t) h w c -> b v t c h w", v=self.num_views)
        B, V, T = curr_obs.shape[:3]
        curr_obs = self.transform_obs(curr_obs, B, V, T)
        curr_obs_encoded = self.image_encoder(curr_obs)  # (B, V, T, D) or similar
        # Optionally fuse with state and text
        if state is not None:
            state_features = self.state_encoder(state)  # (B, D_state)
            # Expand to match sequence length if needed
            state_features = state_features.squeeze(1)  # (B, 1, D_state)
            curr_obs_cond = torch.cat([curr_obs_encoded, state_features], dim=-1)
        else:
            curr_obs_cond = curr_obs_encoded

        if text_embs is not None:
            encoder_hidden_states = torch.cat([curr_obs_cond, text_embs], dim=-1)
        else:
            encoder_hidden_states = curr_obs_cond

        # === 3. Prepare initial noisy action trajectory ===
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
        )

        # === 4. Handle future image conditioning (if used) ===
        next_obs = None
        obs_shape = None
        if self.use_img_denoise:
            obs_t_cont = 0
            obs_t = torch.full(
                size=(batch_size,), fill_value=obs_t_cont, device=device, dtype=dtype
            )
            if self.vision_encoder_type == "dinov3":
                obs_t = obs_t[:, None, None, None, None]      # (B,1,1,1,1)
            else:
                obs_t = obs_t[:, None, None, None, None, None]
            next_obs = torch.randn((batch_size, self.num_views, self.obs_horizon, 3, 224, 224), dtype=dtype, device=device)
            B, V, T = next_obs.shape[:3]
            next_obs = self.transform_obs(next_obs, B, V, T)
            next_obs = self.encode_future_img(next_obs)

            if self.vision_encoder_type == "vjepa2":
                next_obs = rearrange(next_obs, "(b v) t h w c -> b v c t h w", b=B, v=V)
            elif self.vision_encoder_type == "dinov3":
                next_obs = rearrange(next_obs, "(b v t) n c -> b v c t n", b=B, v=V)
            elif self.vision_encoder_type == "vae":
                next_obs = rearrange(next_obs, "(b v t) c h w -> b v c t h w", b=B, v=V)
            if self.vision_encoder_type == "dinov3":
                obs_shape = next_obs.shape[:-1]
            else:
                obs_shape = next_obs.shape[:-2]
        # === 5. Diffusion sampling loop ===
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        def to_tokens(obs):
            if self.glob_len > 0:
                if self.vision_encoder_type == "vjepa2":
                    obs = obs.reshape(*obs_shape, -1)
                local_tokens = self.obs_patchifier(obs[..., -self.patch_len:].reshape(*obs_shape, *self.orig_patch_shape))
                glob_tokens = self.obs_patchifier.proj_glob(
                    obs[..., :-self.patch_len].permute(0, 1, 3, 4, 2).reshape(len(local_tokens), -1, obs.shape[2])
                )
                return torch.cat([glob_tokens, local_tokens], dim=-2)
            else:
                return self.obs_patchifier(obs.reshape(*obs_shape, *self.orig_patch_shape))

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((batch_size,), t_discretized, device=device, dtype=torch.long)

            # Action features
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Future image tokens (if used)
            
            if self.vision_encoder_type == "dinov3":
                obs_t_discretized = (obs_t[:, 0, 0, 0, 0] * self.num_timestep_buckets).long()
            else:
                obs_t_discretized = (obs_t[:, 0, 0, 0, 0, 0] * self.num_timestep_buckets).long()
            if self.use_img_denoise and next_obs is not None:
                noisy_next_obs = to_tokens(next_obs)
                
            # Future tokens
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

            # Position embedding
            if self.config.add_pos_embed:
                if self.use_img_denoise:
                    total_len = action_features.shape[1] + noisy_next_obs.shape[1] + future_tokens.shape[1]
                else:
                    total_len = action_features.shape[1]
                pos_ids = torch.arange(total_len, dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            else:
                pos_embs = None

            # Assemble sequence
            if self.use_img_denoise:
                sa_embs = torch.cat([future_tokens, action_features, noisy_next_obs], dim=1)
            else:
                sa_embs = torch.cat([future_tokens, action_features], dim=1)

            if pos_embs is not None:
                sa_embs = sa_embs + pos_embs

            # Forward through main model
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps_tensor,
                obs_timestep=obs_t_discretized,
                return_all_hidden_states=False,
            )

            # Extract predicted velocity
            if self.use_img_denoise:
                pred_actions = model_output[:, -(actions.shape[1] + self.obs_len) : -self.obs_len]
            else:
                pred_actions = model_output[:, -actions.shape[1]:]

            pred_velocity = self.action_decoder(pred_actions)

            # Euler step
            actions = actions + dt * pred_velocity

        return actions  # (B, action_horizon, action_dim)

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
