from logging import warning
import torch
import torch.nn as nn
import math

# -----------------------------
# 3D Rotary Position Embedding
# -----------------------------
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class Rotary3D(nn.Module):
    """
    3D RoPE: 时间 + 高度 + 宽度
    Input shape: (B, T, H, W, C)
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # 将 dim 平均分配到 T/H/W
        if dim % 3 != 0:
            warning(f"dim % 3 != 0, dim best be divisible by 3 for 3D RoPE, but got {dim}")
        self.dim_h = dim // 3
        self.dim_w = dim // 3
        self.dim_t = dim - self.dim_w - self.dim_h

        # precompute frequencies
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        inv_freq_h = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        self.register_buffer("inv_freq_t", inv_freq_t)
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)

    def forward(self, x, time_interval=None):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # time
        if time_interval is None:
            t_seq = torch.arange(T, device=x.device).float()
        else:
            t_seq = torch.arange(T, device=x.device).float() * time_interval
        freqs_t = torch.einsum("i,j->ij", t_seq, self.inv_freq_t)  # (T, dim_t/2)
        sin_t = freqs_t.sin()
        cos_t = freqs_t.cos()
        sin_t = sin_t.repeat_interleave(2, dim=-1)
        cos_t = cos_t.repeat_interleave(2, dim=-1)
        sin_t = sin_t[:, None, None, :]
        cos_t = cos_t[:, None, None, :]

        # height
        h_seq = torch.arange(H, device=x.device).float()
        freqs_h = torch.einsum("i,j->ij", h_seq, self.inv_freq_h)
        sin_h = freqs_h.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
        cos_h = freqs_h.cos().repeat_interleave(2, dim=-1)[None, :, None, :]

        # width
        w_seq = torch.arange(W, device=x.device).float()
        freqs_w = torch.einsum("i,j->ij", w_seq, self.inv_freq_w)
        sin_w = freqs_w.sin().repeat_interleave(2, dim=-1)[None, None, :, :]
        cos_w = freqs_w.cos().repeat_interleave(2, dim=-1)[None, None, :, :]

        # split channels
        x_t, x_h, x_w = x.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)

        # apply rotary
        x_t = (x_t * cos_t) + (rotate_half(x_t) * sin_t)
        x_h = (x_h * cos_h) + (rotate_half(x_h) * sin_h)
        x_w = (x_w * cos_w) + (rotate_half(x_w) * sin_w)

        x = torch.cat([x_t, x_h, x_w], dim=-1)
        return x


# -----------------------------
# 1D Rotary Position Encoding for action tokens
# -----------------------------
class Rotary1D(nn.Module):
    """
    Standard 1D RoPE for sequences
    x: (B, L, C)
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, time_interval=None):
        # x: (B, L, C)
        B, L, C = x.shape
        if time_interval is None:
            seq = torch.arange(L, device=x.device).float()
        else:
            seq = torch.arange(L, device=x.device).float() * time_interval
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)  # (L, dim/2)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = sin[None, :, :]
        cos = cos[None, :, :]
        x = (x * cos) + (rotate_half(x) * sin)
        return x


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    B = 2
    # 图像 token: 3帧, 4x4 patch, 256维
    img_tokens = torch.randn(B, 3, 4, 4, 256)
    # action token: 21步, 128维
    act_tokens = torch.randn(B, 21, 1536)

    # Linear 映射到统一 channel
    img_linear = nn.Linear(256, 1536)
    img_tokens_mapped = img_linear(img_tokens)

    # RoPE
    rope_3d = Rotary3D(dim=1536)
    rope_1d = Rotary1D(dim=1536)

    img_tokens_encoded = rope_3d(img_tokens_mapped)
    act_tokens_encoded = rope_1d(act_tokens)

    print("Image tokens encoded:", img_tokens_encoded.shape)
    print("Action tokens encoded:", act_tokens_encoded.shape)
