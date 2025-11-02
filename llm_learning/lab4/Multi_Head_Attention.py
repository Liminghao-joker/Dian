# 实现一个基本的Multi-Head Attention层
import torch
from torch import nn
from Self_Attention import dot_production_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # make sure d_model can be divided by num_heads

        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # linear projections
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = q.view(q.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        key = k.view(k.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)

        # attention score
        # x: (batch_size, num_heads, seq_len, d_k)
        # attn: (batch_size, num_heads, seq_len, seq_len)
        x, attn = dot_production_attention(query, key, value, mask)

        # concat heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.Wo(x)
        return x, attn

if __name__ == "__main__":
    # simple test
    torch.manual_seed(22)
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    x = torch.randn(batch_size, seq_len, d_model)
    print("Input:", x)
    # (batch_size, seq_len) -> (batch_size, 1, seq_len) -> (batch_size, seq_len, seq_len)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, seq_len)
    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    out, attn = multi_head_attention(x, x, x, mask)
    print("Output shape:", out.shape)  # (batch_size, seq_len, d_model)
    print("Attention shape:", attn.shape)  # (batch_size, num_heads, seq_len, seq_len)
    print("Output:", out)
    print("\nAttention:", attn)

# Results:
#     Input: tensor([[[ 0.3920,  0.0734, -0.0045, -0.0535, -0.0589,  0.6002,  2.0421,
#            1.3273],
#          [-0.8703, -1.5555, -0.8620, -0.1290,  1.4632,  0.2698, -1.3583,
#           -0.7568],
#          [-0.4102,  0.2939, -0.0538, -0.9547, -1.3138,  0.4306, -1.3356,
#            1.1686],
#          [ 0.9662, -1.7636, -1.9802, -0.0056,  0.8050,  0.7928, -0.6498,
#            0.0352]],

#         [[-1.1281,  0.8134,  0.2734, -0.3833,  1.1319, -0.1240, -0.6294,
#           -0.5749],
#          [-0.8881,  0.7359,  0.6198,  0.3908,  1.0492,  0.3543,  0.0289,
#           -1.0003],
#          [-0.6858,  2.0945,  0.7393, -0.0455, -1.7682, -0.1843, -1.3310,
#            0.5904],
#          [ 0.8605,  0.8002,  1.0861, -0.0626,  1.0136, -0.7909, -0.5482,
#            1.1483]]])
#
# Output shape: torch.Size([2, 4, 8])
# Attention shape: torch.Size([2, 2, 4, 4])
#
# Output: tensor([[[ 1.1834e-01, -3.9756e-02,  8.0518e-02,  1.2401e-01, -4.6627e-02,
#            1.0772e-01, -1.1792e-01, -1.9267e-01],
#          [ 3.2086e-03,  1.1348e-01,  1.9448e-02,  1.1356e-02, -3.3034e-03,
#            2.0603e-01, -1.1484e-01, -5.5605e-02],
#          [ 1.1301e-01,  1.8008e-03,  8.7596e-02,  1.3638e-01, -3.5864e-02,
#            1.4156e-01, -9.3106e-02, -1.8070e-01],
#          [-1.5237e-04,  8.9945e-02,  3.2287e-02,  6.5456e-02, -2.7165e-02,
#            2.1423e-01, -8.5300e-02, -9.2750e-02]],

#         [[-1.5611e-01, -4.3411e-02, -3.7319e-02, -1.8203e-01,  6.1087e-02,
#           -1.4343e-01, -2.1918e-01, -3.2791e-02],
#          [-1.5630e-01, -4.4365e-02, -3.7973e-02, -1.7963e-01,  6.1297e-02,
#           -1.4327e-01, -2.2066e-01, -3.3876e-02],
#          [-1.5492e-01, -3.7424e-02, -3.3151e-02, -1.8411e-01,  6.0784e-02,
#           -1.4350e-01, -2.1553e-01, -3.1258e-02],
#          [-1.5611e-01, -4.3496e-02, -3.7318e-02, -1.6951e-01,  6.2069e-02,
#           -1.4254e-01, -2.2464e-01, -3.7883e-02]]],
#        grad_fn=<UnsafeViewBackward0>)

# Attention: tensor([[[[0.3682, 0.3181, 0.3137, 0.0000],
#           [0.3782, 0.3764, 0.2455, 0.0000],
#           [0.3004, 0.3163, 0.3834, 0.0000],
#           [0.5165, 0.2581, 0.2253, 0.0000]],

#          [[0.2486, 0.4581, 0.2933, 0.0000],
#           [0.4496, 0.1789, 0.3715, 0.0000],
#           [0.3027, 0.4296, 0.2677, 0.0000],
#           [0.4119, 0.2250, 0.3631, 0.0000]]],


#         [[[0.4733, 0.5267, 0.0000, 0.0000],
#           [0.4732, 0.5268, 0.0000, 0.0000],
#           [0.5057, 0.4943, 0.0000, 0.0000],
#           [0.5039, 0.4961, 0.0000, 0.0000]],

#          [[0.4912, 0.5088, 0.0000, 0.0000],
#           [0.4985, 0.5015, 0.0000, 0.0000],
#           [0.4630, 0.5370, 0.0000, 0.0000],
#           [0.5080, 0.4920, 0.0000, 0.0000]]]], grad_fn=<SoftmaxBackward0>)




