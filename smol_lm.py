import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

# creating a vocabulary from the dataset and functions to encode/decode
with open("../shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> List[int]:
    return [char_to_idx[c] for c in s]


def decode(vec: List[int]) -> str:
    return "".join([idx_to_char[i] for i in vec])


# encoding = encode("Hello")
# print(encoding)
# print(decode(encoding))

# Getting the training data
cut = int(len(text) * 0.9)
train_data = encode(text[:cut])
val_data = encode(text[cut:])
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(split: str, batch_size: int, context_length: int):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))

    x = torch.stack([torch.tensor(d[i : i + context_length] for i in ix.tolist())])
    y = torch.stack(
        [torch.tensor(d[i + 1 : i + context_length + 1] for i in ix.tolist())]
    )

    return x.to(device), y.to(device)


# LayerNorm (with Y(weight) and B(bias) params to train)
# class LayerNorm(nn.Module):
#     def __init__(self, dims, eps=1e-5) -> None:
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dims))
#         self.bias = nn.Parameter(torch.zeros(dims))
#
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True, unbiased=False)
#         x_norm = (x - mean) / torch.sqrt(var + self.eps)
#         return x_norm * self.weight + self.bias


# RMSNorm (Root Mean Square Norm)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# RoPE implementation (first calculate all the frequencies to avoid computing them again and again)
def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, base: float = 10000.0
) -> Tuple[Tensor, Tensor]:
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    postions = torch.arange(max_seq_len).float()
    angles = torch.outer(postions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.stack([out1, out2], dim=-1).flatten(-2)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, seq, hd)
        .reshape(b, n_kv * n_rep, seq, hd)
    )


class GQA(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        head_dim: int = 32,
        n_kv_heads: int = 2,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(
            d_model, n_heads * head_dim
        )  # 256 -> 8 * 32 ==> 256 -> 256

        self.k_proj = nn.Linear(
            d_model, n_kv_heads * head_dim
        )  # 256 -> 2 * 32 ==> 256 -> 64 (4 times smaller!!)

        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim)

        self.o_proj = nn.Linear(n_heads * head_dim, d_model)

        self.rope_cos, self.rope_sin = precompute_rope_freqs(head_dim, max_seq_len)

    def forward(self, x: Tensor, head_dim: int) -> Tensor:
        q = self.q_proj(x)  # [b, seq, 256]
        k = self.k_proj(x)  # [b, seq, 64]
        v = self.v_proj(x)  # [b, seq, 64]

        b, seq, _ = x.shape

        q = q.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=0.2, training=self.training)
        out = weights @ v

        # Merge heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq, self.n_heads * self.head_dim)

        return self.o_proj(out)


class SWiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim)
        self.w_up = nn.Linear(d_model, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return F.dropout(self.w_down(gate * up), p=0.2, training=self.training)
