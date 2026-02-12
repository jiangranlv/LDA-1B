from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    RMSNorm
)

from hyper_connections import (
    HyperConnections,
    Residual
)
from diffusers.models.attention import Attention, FeedForward
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

from LDA.model.modules.action_model.flow_matching_head.cdit import TimestepEncoder
from LDA.model.modules.action_model.flow_matching_head.mmdit.mmdit.mmdit_self_attn import JointAttention
from LDA.model.modules.action_model.flow_matching_head.mmdit.mmdit.rope_3d import Rotary3D, Rotary1D
# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm
class DualTimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256 * 2, time_embed_dim=embedding_dim)

    def forward(self, t1, t2):
        dtype = next(self.parameters()).dtype
        timesteps_proj_1 = self.time_proj(t1).to(dtype)
        timesteps_proj_2 = self.time_proj(t2).to(dtype)
        timesteps_proj = torch.cat([timesteps_proj_1, timesteps_proj_2], dim=-1)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,

        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        **kwargs
    ):
        super().__init__()

        # residual functions / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.image_cross_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim)

        self.action_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.action_cross_attn_residual_fn = residual_klass(num_residual_streams, dim = dim)
        self.action_ff_residual_fn = residual_klass(num_residual_streams, dim = dim)

        # pos embedding
        self.positional_embeddings = positional_embeddings
        if positional_embeddings == "sinusoidal":
            self.image_pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
            self.action_pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        elif positional_embeddings == "rope":
            self.image_pos_embed = Rotary3D(dim=dim)
            self.action_pos_embed = Rotary1D(dim=dim)
        else:
            self.image_pos_embed = None
            self.action_pos_embed = None
        # handle optional time conditioning

        dim_gammas = (
            *((dim,) * 4),
            *((dim,) * 4),
        )

        dim_betas = (
            *((dim,) * 2),
            *((dim,) * 2),
        )

        self.cond_dims = (*dim_gammas, *dim_betas)

        to_cond_linear = nn.Linear(dim * 3, sum(self.cond_dims))

        self.to_cond = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            to_cond_linear
        )

        nn.init.zeros_(to_cond_linear.weight)
        nn.init.zeros_(to_cond_linear.bias)
        nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms

        self.image_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        self.image_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        # self.text_attn_layernorm = nn.LayerNorm(cross_attention_dim, elementwise_affine = False)

        self.image_ff_layernorm = nn.LayerNorm(dim, elementwise_affine = False)
        self.action_ff_layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        # attention and feedforward

        self.img_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.action_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # joint self attention
        self.patch_shape = kwargs.get("patch_shape", None)
        self.glob_len = kwargs.get("glob_len", 0)
        self.obs_timesteps = kwargs.get("obs_timesteps", 1)
        self.num_register_tokens = kwargs.get("num_register_tokens", 0)
        self.joint_attn = JointAttention(
            dim_inputs = (dim, dim),
            dim_head = attention_head_dim, 
            heads = num_attention_heads,
            flash = flash_attn,
            patch_shape = self.patch_shape,
            glob_len = self.glob_len,
            obs_timesteps = self.obs_timesteps,
            num_register_tokens=self.num_register_tokens,
        )

        self.image_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,)
        self.action_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        action_tokens,
        text_mask = None,
        time_cond = None,
    ):

        (
            image_pre_attn_gamma,
            image_post_attn_gamma,
            image_pre_ff_gamma,
            image_post_ff_gamma,
            action_pre_attn_gamma,
            action_post_attn_gamma,
            action_pre_ff_gamma,
            action_post_ff_gamma,
            image_pre_attn_beta,
            image_pre_ff_beta,
            action_pre_attn_beta,
            action_pre_ff_beta,
        ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)
        action_tokens, add_action_residual = self.action_attn_residual_fn(action_tokens)

        image_tokens = self.image_attn_layernorm(image_tokens)
        action_tokens = self.action_attn_layernorm(action_tokens)

        image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta
        action_tokens = action_tokens * action_pre_attn_gamma + action_pre_attn_beta

        if self.positional_embeddings == "sinusoidal":
            image_tokens = self.image_pos_embed(image_tokens)
            action_tokens = self.action_pos_embed(action_tokens)

            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
            )
        elif self.positional_embeddings == "rope":
            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
                image_rope_3d_embedding = self.image_pos_embed,
                action_rope_embedding = self.action_pos_embed,
            )
        else:
            # attention
            # 1) self attention
            image_tokens, action_tokens = self.joint_attn(
                inputs = (image_tokens, action_tokens),
            )
        # add attention residual
        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # 2) cross attention
        image_tokens, add_image_residual = self.image_cross_attn_residual_fn(image_tokens)
        action_tokens, add_action_residual = self.action_cross_attn_residual_fn(action_tokens)

        image_tokens = self.image_cross_attn_layernorm(image_tokens)
        action_tokens = self.action_cross_attn_layernorm(action_tokens)
        # text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.img_cross_attn(
            image_tokens,
            encoder_hidden_states=text_tokens,
            attention_mask=text_mask,
        )
        action_tokens = self.action_cross_attn(
            action_tokens,
            encoder_hidden_states=text_tokens,
            attention_mask=text_mask,
        )

        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # condition attention output

        image_tokens = image_tokens * image_post_attn_gamma
        action_tokens = action_tokens * action_post_attn_gamma
        
        # handle feedforward adaptive layernorm
        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        action_tokens, add_action_residual = self.action_ff_residual_fn(action_tokens)
        action_tokens = self.action_ff_layernorm(action_tokens)

        image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta
        action_tokens = action_tokens * action_pre_ff_gamma + action_pre_ff_beta

        # images feedforward

        image_tokens = self.image_ff(image_tokens)
        action_tokens = self.action_ff(action_tokens)
        # images condition feedforward output

        image_tokens = image_tokens * image_post_ff_gamma
        action_tokens = action_tokens * action_post_ff_gamma
        # images feedforward residual

        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # return

        return text_tokens, image_tokens, action_tokens

# mm dit transformer - simply many blocks

class MMDiT(ModelMixin, ConfigMixin):
    @register_to_config 
    def __init__(
        self,
        *,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        final_norm = True,
        num_residual_streams = 1,
        **kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.timestep_encoder = DualTimestepEncoder(self.inner_dim)

        # only norm once for text tokens
        self.text_attn_layernorm = nn.LayerNorm(cross_attention_dim, elementwise_affine = False)
        self.blocks = ModuleList([])

        for _ in range(num_layers):
            block = MMDiTBlock(
                dim = self.inner_dim,
                num_attention_heads = num_attention_heads,
                attention_head_dim = attention_head_dim,
                cross_attention_dim= cross_attention_dim,
                num_residual_streams = num_residual_streams,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                positional_embeddings=positional_embeddings,
                num_positional_embeddings=self.config.max_num_positional_embeddings,
                final_dropout=final_dropout,
                **kwargs
            )

            self.blocks.append(block)

        self.norm = RMSNorm(self.inner_dim) if final_norm else nn.Identity()
        self.action_norm = RMSNorm(self.inner_dim) if final_norm else nn.Identity()

        # Output blocks
        self.action_proj_out = nn.Linear(self.inner_dim, self.config.output_dim)
        self.image_proj_out = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(    
        self,
        *,
        image_tokens,
        action_tokens,
        text_tokens,
        register_tokens = None,
        text_mask = None,
        ada_cond = None,
        action_t = None,
        obs_t = None,
    ):

        if register_tokens is not None:
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')
        image_tokens = self.expand_streams(image_tokens)
        action_tokens = self.expand_streams(action_tokens)
        
        text_tokens = self.text_attn_layernorm(text_tokens)
        # cond embedding
        time_cond = self.timestep_encoder(action_t, obs_t)
        if ada_cond is not None:
            time_cond = torch.cat([ada_cond, time_cond], dim=-1)

        for ind, block in enumerate(self.blocks):

            text_tokens, image_tokens, action_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                action_tokens = action_tokens,
                text_mask = text_mask,
            )
        if register_tokens is not None:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        image_tokens = self.reduce_streams(image_tokens)
        action_tokens = self.reduce_streams(action_tokens)

        image_tokens = self.norm(image_tokens)
        action_tokens = self.action_norm(action_tokens)
        
        # proj to output dim
        action_tokens = self.action_proj_out(action_tokens)
        image_tokens = self.image_proj_out(image_tokens)

        return image_tokens, action_tokens

def test_mmdit():
    device = "cpu"
    batch_size = 2
    
    # Dimensions
    dim_text = 384
    inner_dim = 512  # 8 heads * 64 dim
    num_img_tokens = 64
    num_action_tokens = 10
    num_text_tokens = 16
    num_register_tokens = 4
    # Model (with time conditioning and register tokens)
    model = MMDiT(
        num_attention_heads=8,
        attention_head_dim=64,
        cross_attention_dim=dim_text,
        num_layers=2,
        dropout=0.1,
        activation_fn="gelu",
        attention_bias=True,
        norm_elementwise_affine=False,
        final_norm=True,
        num_residual_streams=1,
    ).to(device)
    
    # Inputs
    text_tokens = torch.randn(batch_size, num_text_tokens, dim_text).to(device)
    image_tokens = torch.randn(batch_size, num_img_tokens, inner_dim).to(device)
    action_tokens = torch.randn(batch_size, num_action_tokens, inner_dim).to(device)
    register_tokens = torch.randn(batch_size, num_register_tokens, inner_dim).to(device)

    time_cond = torch.randn(batch_size, ).to(device)  

    
    # Forward pass
    image_out, action_out = model(
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        action_tokens=action_tokens,
        time_cond=time_cond,  
        register_tokens=register_tokens,
        text_mask=None,
    )
    
    print("✅ Success!")
    print(f"Image output: {image_out.shape}")   # [2, 64, 512]
    print(f"Action output: {action_out.shape}") # [2, 10, 512]
    
    # Sanity check
    assert not torch.isnan(image_out).any()
    assert not torch.isnan(action_out).any()

if __name__ == "__main__":
    test_mmdit()