import torch
import torch.nn as nn
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, d_model, weights = None, eps = 1e-5):
        super().__init__()
        self.d_model = d_model
        if weights is None:
            self.weights = nn.Parameter(torch.ones(d_model))
        else:
            self.weights = weights
        self.eps = eps

    def forward(self, activation):
        norm_factor = torch.sqrt(torch.mean(activation**2, dim=-1, keepdim=True) + self.eps)
        return activation / norm_factor * self.weights

if __name__ == '__main__':
    rmsnorm = RMSNorm(512)
    ac = torch.randn(512)
    print(rmsnorm(ac))

class GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x/np.sqrt(2)))

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.GELU = GELU()

    def forward(self, x):
        x = self.w1(x)
        x = self.GELU(x)
        x = self.w2(x)
        return x

def softmax(t : torch.FloatTensor, dim: int):
    exp_tensor = torch.exp(t - torch.max(t, dim=dim, keepdim=True).values)
    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask, pdrop):
    qk = Q @ K
