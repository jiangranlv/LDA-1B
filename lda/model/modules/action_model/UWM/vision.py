from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torchvision.transforms import Normalize


def get_imagenet_norm(inplace=True):
    """
    Construct an ImageNet normalization transform.
    """
    return Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=inplace
    )


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Recursively replace submodules that satisfy a given predicate.

    Args:
        root_module (nn.Module): The root module to process.
        predicate (Callable[[nn.Module], bool]): A function that takes a module as input and
            returns True if the module should be replaced.
        func (Callable[[nn.Module], nn.Module]): A function that takes a module as input and
            returns a new module to replace it.
        **kwargs: Additional keyword arguments to be passed to the ResNet model constructor.
    """
    if predicate(root_module):
        return func(root_module)

    # Replace all submodules that satisfy the predicate
    module_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in module_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # Verify that all submodules are replaced
    module_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(module_list) == 0
    return root_module


def get_resnet(name, embed_dim, weights=None, replace_batch_norm=True, **kwargs):
    """
    Construct a ResNet model with a custom output embedding dimension and optional batch norm replacement.

    Args:
        name (str): The name of the ResNet architecture to use (e.g., "resnet18", "resnet34", "resnet50").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        replace_batch_norm (bool, optional): If True, replaces `nn.BatchNorm2d` layers with `nn.GroupNorm` layers. Default is True.
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)
    if replace_batch_norm:
        resnet = replace_submodules(
            root_module=resnet,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // 16,
                num_channels=x.num_features,
            ),
        )
    return resnet


def get_vit(name, embed_dim, weights=None, **kwargs):
    """
    Construct a Vision Transformer (ViT) model with a custom output embedding dimension.

    Args:
        name (str): The name of the ViT architecture to use (e.g., "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        **kwargs: Additional keyword arguments to be passed to the ViT model constructor.
    """
    func = getattr(torchvision.models, name)
    vit = func(weights=weights, **kwargs)
    vit.heads = nn.Linear(768, embed_dim)
    return vit


def get_clip(embed_dim, **kwargs):
    """
    Construct a pretrained CLIP encoder with a custom output embedding dimension.

    Args:
        embed_dim (int): The dimension of the output embedding.
        **kwargs: Additional keyword arguments to be passed to the timm model creation function.
    """
    clip = timm.create_model(
        "hf_hub:timm/vit_base_patch32_clip_224.openai", pretrained=True, **kwargs
    )
    clip.head = nn.Linear(768, embed_dim)
    return clip


class ResNetImageEncoder(nn.Module):
    """
    Multi-view image encoder using a ResNet backbone.

    The input is expected to be a tensor with shape (B, V, C, T, H, W), where:
      - B is the batch size,
      - V is the number of views,
      - C is the number of channels,
      - T is the number of frames,
      - H and W are the image height and width, respectively.
    The encoder reshapes the input to treat each view and frame as an individual image,
    extracts features using the ResNet model, and then concatenates the features across
    all views and frames.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_resnet("resnet18", embed_dim, weights="IMAGENET1K_V1")

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        feats = self.model(self.norm(imgs))
        feats = rearrange(feats, "(b v t) c -> b (v t c)", b=B, v=V)
        return feats


class ViTImageEncoder(nn.Module):
    """
    Multi-view image encoder using a Vision Transformer (ViT) backbone.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.model = get_vit("vit_b_32", embed_dim, weights="IMAGENET1K_V1")

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v t c h w -> (b v t) c h w")

        # Reshape and permute the input tensor
        x = self.model._process_input(imgs)

        # Expand the class token to the full batch
        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        # Get raw tokens
        x = self.model.encoder(x)
        x = self.model.heads(x[:, 0])
        feats = rearrange(x, "(b v t) c -> b (v t c)", b=B, v=V)
        return feats  # (b, v*t*c)


class ViTImagePatchEncoder(nn.Module):
    """
    Multi-view image patch encoder using a Vision Transformer (ViT) backbone with learnable positional embeddings.

    Args:
        num_views (int): Number of camera views in the input.
        num_frames (int): Number of frames per view.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, num_frames: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_vit("vit_b_32", embed_dim, weights="IMAGENET1K_V1")

        # Learnable embeddings
        self.pos_shift = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )
        self.pos_scale = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v t c h w -> (b v t) c h w")
        imgs = self.norm(imgs)

        # Reshape and permute the input tensor
        x = self.model._process_input(imgs)

        # Expand the class token to the full batch
        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        # Get raw tokens
        x = self.model.encoder(x)
        x = self.model.heads(x)

        # Add learned positional embeddings
        feats = rearrange(x, "(b v t) n c -> b (v t) n c", b=B, v=V)
        feats = feats * (1 + self.pos_scale) + self.pos_shift
        return feats.flatten(1, 2)  # (b, v*t*n, c)
