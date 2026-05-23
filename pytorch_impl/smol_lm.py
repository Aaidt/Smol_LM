import torch
import torch.nn as nn
from typing import List

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


# LayerNorm (with Y(gamma) and B(beta) params to train)


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
