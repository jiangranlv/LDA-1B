from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

from hyper_connections import (
    HyperConnections,
    Residual
)

from lda.model.modules.action_model.flow_matching_head.cdit import TimestepEncoder
# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

from dataclasses import dataclass

@dataclass
class TokenLayout:
    register: tuple[int, int]
    curr_obs: tuple[int, int]
    next_obs: tuple[int, int]
    action: tuple[int, int]
    state: tuple[int, int]
def print_attention_ratio(attn, layout, name=""):
    """
    attn: Tensor [B, H, Q, K]
    """
    B, H, Q, K = attn.shape

    q0, q1 = layout.action
    action_attn = attn[:, :, q0:q1, :]   # [B, H, Qa, K]

    def region_ratio(k_range):
        k0, k1 = k_range
        return action_attn[..., k0:k1].sum(dim=-1).mean().item()

    ratios = {
        "register": region_ratio(layout.register),
        "curr_obs": region_ratio(layout.curr_obs),
        "next_obs": region_ratio(layout.next_obs),
        "action":   region_ratio(layout.action),
        "state": region_ratio(layout.state)
    }

    total = sum(ratios.values())
    print(f"\n[Attention Ratio] {name}")
    for k, v in ratios.items():
        print(f"  {k:10s}: {v / total:.3f}")

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict(),
        patch_shape = None,
        glob_len = 0,
        obs_timesteps = 1,
        num_register_tokens = 0,
        save_attn = False,
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
        
        # 3d rope related 
        self.patch_shape = patch_shape
        self.glob_len = glob_len
        self.obs_timesteps = obs_timesteps
        self.num_register_tokens = num_register_tokens

        self.save_attn = save_attn

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def get_image_patch_tokens(self, obs):
        '''
        Get image tokens which has shape (B, T, H, W, D) from (B, T*(H*W+G), D)
        '''
        B, N, D = obs.shape
        H, W = self.patch_shape
        T = self.obs_timesteps
        obs_wo_register = obs[:, self.num_register_tokens:, :]  # remove register tokens
        register_tokens = obs[:, :self.num_register_tokens, :]  # (B, num_register_tokens, D)
        img_tokens = obs_wo_register.reshape(B, T, -1, D)
        image_patch_part = img_tokens[:, :, self.glob_len:, :]
        glob_part = img_tokens[:, :, :self.glob_len, :]
        return image_patch_part.reshape(B, T, H, W, D), glob_part, register_tokens

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None,
        image_rope_3d_embedding = None,
        image_time_interval = 0.5,
        action_rope_embedding = None,
        action_time_interval = 0.1,
    ):
        '''
        inputs: tuple, contain multi modality inputs
        '''

        token_lens = [x.shape[1] for x in inputs]


        device = self.dummy.device
        
        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for i, (x, mask, to_qkv, q_rmsnorm, k_rmsnorm) in enumerate(zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms)):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            if image_rope_3d_embedding is not None and i == 0: # i == 0 means image modality
                q, k, v = qkv
                B, num_heads, N, head_dim  = q.shape
                image_part_q, glob_part_q, register_tokens_q = self.get_image_patch_tokens(q.reshape(B, N, -1))
                image_part_k, glob_part_k, register_tokens_k = self.get_image_patch_tokens(k.reshape(B, N, -1))
                # B, T, H, W, D -> B, T*(H*W), D
                image_part_q = image_rope_3d_embedding(image_part_q, time_interval=image_time_interval).reshape(B, self.obs_timesteps, -1, head_dim * num_heads)
                image_part_k = image_rope_3d_embedding(image_part_k, time_interval=image_time_interval).reshape(B, self.obs_timesteps,-1, head_dim * num_heads)
                # concat glob token and image part token
                dino_tokens_q = torch.cat((glob_part_q, image_part_q), dim=2).reshape(B, -1, head_dim * num_heads)
                dino_tokens_k = torch.cat((glob_part_k, image_part_k), dim=2).reshape(B, -1, head_dim * num_heads)
                q = torch.cat((register_tokens_q, dino_tokens_q), dim=1).reshape(B, num_heads, N, head_dim)
                k = torch.cat((register_tokens_k, dino_tokens_k), dim=1).reshape(B, num_heads, N, head_dim)
                qkv = torch.stack((q, k, v))

            if action_rope_embedding is not None and i == 1: # i == 1 means action modality
                q, k, v = qkv
                B, num_heads, N, head_dim  = q.shape
                q = action_rope_embedding(q.reshape(B, N, -1), time_interval=action_time_interval).reshape(B, num_heads, N, head_dim)
                k = action_rope_embedding(k.reshape(B, N, -1), time_interval=action_time_interval).reshape(B, num_heads, N, head_dim)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, intermeds = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_image,
        dim_action,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # residual functions / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_text)
        self.text_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_text)

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_image)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_image)

        self.action_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_action)
        self.action_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_action)
        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (
                *((dim_text,) * 4),
                *((dim_image,) * 4),
                *((dim_action,) * 4),
            )

            dim_betas = (
                *((dim_text,) * 2),
                *((dim_image,) * 2),
                *((dim_action,) * 2),
            )

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)
        self.action_attn_layernorm = nn.LayerNorm(dim_action, elementwise_affine = not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)
        self.action_ff_layernorm = nn.LayerNorm(dim_action, elementwise_affine = not has_cond)

        # attention and feedforward

        self.joint_attn = JointAttention(
            dim_inputs = (dim_text, dim_image, dim_action),
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.image_ff = FeedForward(dim_image, **ff_kwargs)
        self.action_ff = FeedForward(dim_action, **ff_kwargs)
    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        action_tokens,
        text_mask = None,
        time_cond = None,
        skip_feedforward_text_tokens = True
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_post_ff_gamma,
                action_pre_attn_gamma,
                action_post_attn_gamma,
                action_pre_ff_gamma,
                action_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                image_pre_attn_beta,
                image_pre_ff_beta,
                action_pre_attn_beta,
                action_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)
        action_tokens, add_action_residual = self.action_attn_residual_fn(action_tokens)

        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)
        action_tokens = self.action_attn_layernorm(action_tokens)
        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta
            action_tokens = action_tokens * action_pre_attn_gamma + action_pre_attn_beta

        # attention

        text_tokens, image_tokens, action_tokens = self.joint_attn(
            inputs = (text_tokens, image_tokens, action_tokens),
            masks = (text_mask, None)
        )

        # condition attention output

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            image_tokens = image_tokens * image_post_attn_gamma
            action_tokens = action_tokens * action_post_attn_gamma
        # add attention residual

        text_tokens = add_text_residual(text_tokens)
        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        
        # handle feedforward adaptive layernorm

        if not skip_feedforward_text_tokens:
            text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
            text_tokens = self.text_ff_layernorm(text_tokens)

            if self.has_cond:
                text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        action_tokens, add_action_residual = self.action_ff_residual_fn(action_tokens)
        action_tokens = self.action_ff_layernorm(action_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta
            action_tokens = action_tokens * action_pre_ff_gamma + action_pre_ff_beta

        # images feedforward

        image_tokens = self.image_ff(image_tokens)
        action_tokens = self.action_ff(action_tokens)
        # images condition feedforward output

        if self.has_cond:
            image_tokens = image_tokens * image_post_ff_gamma
            action_tokens = action_tokens * action_post_ff_gamma
        # images feedforward residual

        image_tokens = add_image_residual(image_tokens)
        action_tokens = add_action_residual(action_tokens)
        # early return, for last block in mmdit

        if skip_feedforward_text_tokens:
            return text_tokens, image_tokens, action_tokens

        # text feedforward

        text_tokens = self.text_ff(text_tokens)

        # text condition feedforward output

        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma

        # text feedforward residual

        text_tokens = add_text_residual(text_tokens)

        # return

        return text_tokens, image_tokens, action_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_image,
        dim_action, 
        dim_cond = None,
        num_register_tokens = 0,
        final_norm = True,
        num_residual_streams = 4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.has_register_tokens = num_register_tokens > 0
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_image))
        nn.init.normal_(self.register_tokens, std = 0.02)

        if dim_cond is not None:
            self.timestep_encoder = TimestepEncoder(dim_cond)

        self.blocks = ModuleList([])

        for _ in range(depth):
            block = MMDiTBlock(
                dim_image = dim_image,
                dim_action = dim_action,
                dim_cond = dim_cond,
                num_residual_streams = num_residual_streams,
                **block_kwargs
            )

            self.blocks.append(block)

        self.norm = RMSNorm(dim_image) if final_norm else nn.Identity()
        self.action_norm = RMSNorm(dim_action) if final_norm else nn.Identity()

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        action_tokens,
        text_mask = None,
        time_cond = None,
        task_embedding = None,
        should_skip_last_feedforward = True
    ):

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = image_tokens.shape[0])
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')

        text_tokens = self.expand_streams(text_tokens)
        image_tokens = self.expand_streams(image_tokens)
        action_tokens = self.expand_streams(action_tokens)
        # cond embedding
        if time_cond is not None:
            time_cond = self.timestep_encoder(time_cond)
            if task_embedding is not None:
                time_cond += task_embedding

        for ind, block in enumerate(self.blocks):
            is_last = ind == (len(self.blocks) - 1)

            text_tokens, image_tokens, action_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                action_tokens = action_tokens,
                text_mask = text_mask,
                skip_feedforward_text_tokens = is_last and should_skip_last_feedforward
            )

        if self.has_register_tokens:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        text_tokens = self.reduce_streams(text_tokens)
        image_tokens = self.reduce_streams(image_tokens)
        action_tokens = self.reduce_streams(action_tokens)

        image_tokens = self.norm(image_tokens)
        action_tokens = self.action_norm(action_tokens)

        return text_tokens, image_tokens, action_tokens
