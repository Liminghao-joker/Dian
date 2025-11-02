# 实现一个基础的self_attention层

import torch
from torch import nn
import math

# dot product attention
def dot_production_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None):
    # q, k, v: (batch, seq_len, d) or (batch, num_heads, seq_len, d)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]) # (batch, num_heads, seq_len, seq_len)
    if mask is not None:
        # 若 mask 没有head的维度，则扩展一维
        if mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # (batch, seq_len, d)
    return out, attn

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_k:int) -> None:  # d_k: hidden size
        super().__init__()
        self.Wq = nn.Linear(d_k, d_k, bias=False)
        self.Wk = nn.Linear(d_k, d_k, bias=False)
        self.Wv = nn.Linear(d_k, d_k, bias=False)
        self.Wo = nn.Linear(d_k, d_k, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        # x (batch_size, seq_len, d)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        out, attn = dot_production_attention(q, k, v, mask)
        # q, k, v (batch_size, seq_len, d)
        out = self.Wo(out)
        return out, attn


if __name__ == "__main__":
    # simple test
    torch.manual_seed(22)
    batch_size = 2
    seq_len = 4
    d_k = 8
    x = torch.randn(batch_size, seq_len, d_k)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, seq_len)
    self_attention = SelfAttentionBlock(d_k)
    out, attn = self_attention(x, mask)
    print("Output shape:", out.shape)  # (batch_size, seq_len, d_k)
    print("Attention shape:", attn.shape)  # (batch_size, seq_len, seq_len)
    print("Output:", out)
    print("Attention:", attn)
    print("Test passed!")