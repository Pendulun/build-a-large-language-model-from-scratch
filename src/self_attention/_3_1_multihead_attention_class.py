import torch
import torch.nn as nn

from src.self_attention._1_1_self_attention_class import CausalSelfAttention_v2


class MultiHeadAttentionWrapper_v1(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalSelfAttention_v2(d_in, d_out, context_length, dropout,
                                   qkv_bias) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention_v2(nn.Module):
    """
    Versão mais performática que a v1 ao não aplicar sequencialmente cada
    módulo de self-attention.
    """

    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # A dimensão de saída de cada head é proporcional ao d_out
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Projeção apenas para combinar as saídas de cada head
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Cada um desses tensores tem shape (b, num_tokens, d_out)
        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        # View é um reshape dos tensores. Como d_out = num_heads * heads_dim,
        # ele vai criar tensores novos 4-dimensionais para facilitar as operações
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transforma em (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, num_heads, num_tokens, head_dim) X (b, num_heads, head_dim, num_tokens)
        # e isso funciona
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec: torch.Tensor = (attn_weights @ values).transpose(1, 2)
        # Volta para o shape correto
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
