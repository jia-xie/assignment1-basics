import torch
import math
from einops import rearrange, reduce


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        mean = 0
        std = math.sqrt(2 / (out_feature + in_feature))
        w = torch.empty((out_feature, in_feature), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=-3 * std, b=3 * std)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Emdebbing(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        mean = 0
        std = 1
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=-2, b=3)
        self.weight = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)
        result = x * self.weight / rms
        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = a1 * torch.sigmoid(a1)
        return self.w2(silu * self.w3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device)
        i = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (i / d_k))
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_rot = self.cos[token_positions]
        sin_rot = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos_rot - x_odd * sin_rot
        x_odd_rot = x_even * sin_rot + x_odd * cos_rot

        out = torch.empty_like(x)
        out[..., 0::2] = x_even_rot
        out[..., 1::2] = x_odd_rot

        return out


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
):
    d_k = Q.shape[-1]
    attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    attention_scores = torch.where(mask, attention_scores, float("-inf"))
    attention_weights = softmax(attention_scores)
    return attention_weights @ V


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, rope=None, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = Linear(d_model, 3 * d_model, device, dtype)
        self.out = Linear(d_model, d_model, device, dtype)

        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        if token_positions is None:
            token_positions = torch.arange(T, device=x.device)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = ~torch.triu(
            torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1
        )

        y = scaled_dot_product_attention(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out(y)
    

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rope = rope
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))