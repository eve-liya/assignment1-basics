from typing import Optional
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

def softmax(t: torch.FloatTensor, dim: int):
    exp_tensor = torch.exp(t - torch.max(t, dim=dim, keepdim=True).values)
    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor, mask: Optional[torch.BoolTensor], pdrop: Optional[float] = None):
    d_k = K.shape[-1]
    attention_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask, float('-inf')) 

    attention_weights = softmax(attention_scores, dim=-1)

    if pdrop is not None:
        attention_weights = nn.functional.dropout(attention_weights, p=pdrop, training=True)  # Apply dropout

    return attention_weights @ V

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, weights: dict[str, torch.FloatTensor] | None, attn_pdrop: float = 0):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.attn_pdrop = attn_pdrop

        self.Q_proj = nn.Linear(d_model, d_model, bias=False)
        self.K_proj = nn.Linear(d_model, d_model, bias=False)
        self.V_proj = nn.Linear(d_model, d_model, bias=False)
        self.O_proj = nn.Linear(d_model, d_model, bias=False)
        if weights is not None:
            with torch.no_grad():
                self.Q_proj.weight.copy_(torch.stack([weights[f"q_heads.{row}.weight"] for row in range(num_heads)]).reshape(d_model, d_model))
                self.K_proj.weight.copy_(torch.stack([weights[f"k_heads.{row}.weight"] for row in range(num_heads)]).reshape(d_model, d_model))
                self.V_proj.weight.copy_(torch.stack([weights[f"v_heads.{row}.weight"] for row in range(num_heads)]).reshape(d_model, d_model))
                self.O_proj.weight.copy_(weights['output_proj.weight'])



    def forward(self, x: torch.FloatTensor):
        batch_size, seq_len, _ = x.shape

        Q = self.Q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # (B, T, H, d_k)
        K = self.K_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # (B, T, H, d_k)
        V = self.V_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # (B, T, H, d_v)

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        attention = scaled_dot_product_attention(Q, K, V, mask, self.attn_pdrop)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.O_proj(attention)
        return output

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float | None = None, residual_pdrop: float | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.multi_head_self_attn = Multihead_self_attention(d_model, num_heads, None, attn_pdrop)
        self.drop1 = nn.Dropout(residual_pdrop)

        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor):
        normalized_attn = self.norm1(x)
        attn = self.multi_head_self_attn(normalized_attn)
        x = x + self.drop1(attn)

        normalized_ffn = self.norm2(x)
        ffn = self.ffn(normalized_ffn)
        return x + self.drop2(ffn)

