import torch
from src.self_attention._1_1_self_attention_class import CausalSelfAttention_v1, CausalSelfAttention_v2


def compute_all_context_embeddings_v1():
    inputs = _get_input_embeddings()
    torch.manual_seed(123)
    d_in = inputs.shape[1]
    d_out = 2
    ca = CausalSelfAttention_v1(d_in, d_out, qkv_bias=False)
    print(ca(inputs))


def compute_all_context_embeddings_v2():
    inputs = _get_input_embeddings()
    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    d_in = inputs.shape[1]
    d_out = 2
    ca = CausalSelfAttention_v2(d_in, d_out, context_length, 0.0)
    print(ca(batch))


def _get_input_embeddings() -> torch.Tensor:
    """
    Esses embeddings viriam após a etapa de positional encoding
    """
    return torch.tensor([
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55]  # step (x^6)
    ])


if __name__ == "__main__":
    compute_all_context_embeddings_v1()
    compute_all_context_embeddings_v2()
