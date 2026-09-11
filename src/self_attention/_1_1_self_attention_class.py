import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        # Essas matrizes serão aprendidas
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        # Scaled-dot product
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Outra versão do Self-Attention que usa nn.Linear ao invés de nn.Parameter
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Essas matrizes serão aprendidas
        # O nn.Linear possui um método otimizado de inicialização
        # de valores que contribui opara o treinamento do modelo
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        # Scaled-dot product
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalSelfAttention_v1(nn.Module):
    """
    Módulo de Self-Attention que não leva em consideração os tokens
    que vem após o token sendo analisado
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Essas matrizes serão aprendidas
        # O nn.Linear possui um método otimizado de inicialização
        # de valores que contribui opara o treinamento do modelo
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores: torch.Tensor = queries @ keys.T

        # Definimos todos os elementos da diagonal superior de attn_scores
        # como -inf, o que vai fazer com que, ao passarem pelo softmax, virem 0
        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length),
                          diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        # Scaled-dot product
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec


class CausalSelfAttention_v2(nn.Module):
    """
    Extensão do CausalSelfAttention_v1 que também implementa
    dropout sobre os attention-weights para prevenir o overfitting
    ao garantir que o modelo não fique dependente de nenhum padrão
    muito específico encontrado durante o treinamento.
    Essa classe também leva em consideração o processamento em batch
    recebendo como entrada um tensor 3-dimensional
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Isso permite que a máscara seja transferida para a gpu se a instância
        # dessa classe for enviada para a GPU. Se não, self.mask seria apenas um
        # torch.Tensor que NÃO É automaticamente transferido para a gpu assim como
        # as outras classes acima.
        # Isso só está sendo usado pq queremos definir a mask como self.mask. Se es-
        # tivéssemos fazendo como a v1 definindo a mask dentro de feedforward(), isso não
        # seria necessário. Entretanto, como é mais performático criar a máscara apenas uma
        # vez, isso tudo é necessário.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x é um tensor 3-dimensional
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores: torch.Tensor = queries @ keys.transpose(1, 2)
        # No pytorch, métodos terminados com "_" aplicam a operação inplace
        # evitando cópias desnecessárias
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens],
                                 -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # O dropout é aplicado
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
