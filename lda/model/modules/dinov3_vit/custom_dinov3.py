# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from safetensors import safe_open


class DINOv3ViTConfig:
    """DINOv3ViT配置类 - 基于HuggingFace config.json"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.architectures = config_dict.get("architectures", ["DINOv3ViTModel"])
        self.attention_dropout = config_dict.get("attention_dropout", 0.0)
        self.drop_path_rate = config_dict.get("drop_path_rate", 0.0)
        self.hidden_act = config_dict.get("hidden_act", "gelu")
        self.hidden_size = config_dict.get("hidden_size", 384)
        self.image_size = config_dict.get("image_size", 224)
        self.initializer_range = config_dict.get("initializer_range", 0.02)
        self.intermediate_size = config_dict.get("intermediate_size", 1536)
        self.key_bias = config_dict.get("key_bias", False)
        self.layer_norm_eps = config_dict.get("layer_norm_eps", 1e-5)
        self.layerscale_value = config_dict.get("layerscale_value", 1.0)
        self.mlp_bias = config_dict.get("mlp_bias", True)
        self.model_type = config_dict.get("model_type", "dinov3_vit")
        self.num_attention_heads = config_dict.get("num_attention_heads", 6)
        self.num_channels = config_dict.get("num_channels", 3)
        self.num_hidden_layers = config_dict.get("num_hidden_layers", 12)
        self.num_register_tokens = config_dict.get("num_register_tokens", 4)
        self.patch_size = config_dict.get("patch_size", 16)
        self.pos_embed_jitter = config_dict.get("pos_embed_jitter", None)
        self.pos_embed_rescale = config_dict.get("pos_embed_rescale", 2.0)
        self.pos_embed_shift = config_dict.get("pos_embed_shift", None)
        self.proj_bias = config_dict.get("proj_bias", True)
        self.query_bias = config_dict.get("query_bias", True)
        self.rope_theta = config_dict.get("rope_theta", 100.0)
        self.torch_dtype = config_dict.get("torch_dtype", "float32")
        self.use_gated_mlp = config_dict.get("use_gated_mlp", False)
        self.value_bias = config_dict.get("value_bias", True)
        
        # 计算embed_dim
        self.embed_dim = self.hidden_size


class DINOv3ViTEmbeddings(nn.Module):
    """DINOv3ViT嵌入层"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        
        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=config.initializer_range)
        nn.init.trunc_normal_(self.register_tokens, std=config.initializer_range)
        nn.init.zeros_(self.mask_token)
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)  # (B, hidden_size, H, W)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # 添加CLS token和register tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)
        return embeddings


class DINOv3ViTAttention(nn.Module):
    """DINOv3ViT注意力层"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.query_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.key_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.value_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.proj_bias)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        return attn_output


class DINOv3ViTMLP(nn.Module):
    """DINOv3ViT MLP层"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        
        if config.use_gated_mlp:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        
        self.activation_fn = nn.GELU()
        
    def forward(self, hidden_states):
        if self.config.use_gated_mlp:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            hidden_states = self.activation_fn(gate) * up
        else:
            hidden_states = self.up_proj(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
        
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class DINOv3ViTLayerScale(nn.Module):
    """DINOv3ViT层缩放"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(config.hidden_size) * config.layerscale_value)
        
    def forward(self, hidden_states):
        return hidden_states * self.lambda1


class DINOv3ViTLayer(nn.Module):
    """DINOv3ViT层"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        
        self.attention = DINOv3ViTAttention(config)
        self.mlp = DINOv3ViTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)
        
    def forward(self, hidden_states):
        # 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP + 残差连接
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DINOv3ViTModel(nn.Module):
    """DINOv3ViT模型 - 基于源码实现"""
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = DINOv3ViTEmbeddings(config)
        
        # Transformer层
        self.layers = nn.ModuleList([
            DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DINOv3ViTEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
            if module.config.num_register_tokens > 0:
                module.register_tokens.data = nn.init.trunc_normal_(
                    module.register_tokens.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.register_tokens.dtype)
            module.mask_token.data.zero_()
        elif isinstance(module, DINOv3ViTLayerScale):
            module.lambda1.data.fill_(self.config.layerscale_value)
    
    def forward(self, pixel_values):
        # 嵌入
        hidden_states = self.embeddings(pixel_values)
        
        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 分离CLS token和patch tokens
        cls_token = hidden_states[:, 0]  # CLS token
        patch_tokens = hidden_states[:, 1+self.config.num_register_tokens:]  # patch tokens
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': cls_token,
            'patch_tokens': patch_tokens
        }


def load_dinov3_from_checkpoint(checkpoint_path: str) -> DINOv3ViTModel:
    """从checkpoint加载DINOv3ViTModel"""
    
    # 加载配置
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = DINOv3ViTConfig(config_dict)
    
    # 创建模型
    model = DINOv3ViTModel(config)
    
    # 加载权重
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        # 使用safetensors加载
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # 尝试加载pytorch_model.bin
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No weights file found in {checkpoint_path}")
    
    # 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return model.eval()


def load_dinov3_from_pth(pth_path: str, model_size: str = "s") -> DINOv3ViTModel:
    """从.pth文件加载DINOv3ViTModel"""
    
    # 根据模型大小创建配置
    size_to_config = {
        "s": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "b": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "l": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32},
    }
    
    model_config = size_to_config.get(model_size, size_to_config["s"])
    
    # 创建配置
    config_dict = {
        "hidden_size": model_config["hidden_size"],
        "num_hidden_layers": model_config["num_hidden_layers"],
        "num_attention_heads": model_config["num_attention_heads"],
        "patch_size": 16,
        "num_channels": 3,
        "num_register_tokens": 4,
        "intermediate_size": model_config["hidden_size"] * 4,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
        "layerscale_value": 1.0,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "use_gated_mlp": False,
    }
    
    config = DINOv3ViTConfig(config_dict)
    
    # 创建模型
    model = DINOv3ViTModel(config)
    
    # 加载权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # 转换权重格式
    converted_state_dict = convert_dinov3_state_dict(checkpoint)
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return model.eval()


def convert_dinov3_state_dict(state_dict):
    """转换DINOv3原始格式的state_dict"""
    import re
    
    converted_dict = {}
    
    # 定义key映射规则
    key_mappings = {
        r"cls_token": r"embeddings.cls_token",
        r"mask_token": r"embeddings.mask_token", 
        r"storage_tokens": r"embeddings.register_tokens",
        r"patch_embed.proj": r"embeddings.patch_embeddings",
        r"blocks.(\d+).attn.proj": r"layers.\1.attention.o_proj",
        r"blocks.(\d+).attn.qkv": r"layers.\1.attention.qkv_proj",
        r"blocks.(\d+).mlp.fc1": r"layers.\1.mlp.up_proj",
        r"blocks.(\d+).mlp.fc2": r"layers.\1.mlp.down_proj",
        r"blocks.(\d+).norm1": r"layers.\1.norm1",
        r"blocks.(\d+).norm2": r"layers.\1.norm2",
        r"blocks.(\d+).ls1.gamma": r"layers.\1.layer_scale1.lambda1",
        r"blocks.(\d+).ls2.gamma": r"layers.\1.layer_scale2.lambda1",
        r"norm": r"norm",
    }
    
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        # 应用正则表达式替换
        for pattern, replacement in key_mappings.items():
            new_key = re.sub(pattern, replacement, new_key)
        
        # 处理特殊情况
        if "bias_mask" in old_key or "attn.k_proj.bias" in old_key or "local_cls_norm" in old_key:
            continue
        if "embeddings.mask_token" in new_key:
            tensor = tensor.unsqueeze(1)
            
        converted_dict[new_key] = tensor
        
    return converted_dict
