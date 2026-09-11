from timeit import default_timer as timer
import torch
from src.self_attention._3_1_multihead_attention_class import MultiHeadAttentionWrapper_v1, MultiHeadAttention_v2


def compute_all_context_embeddings_v1():
    inputs = _get_input_embeddings()
    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    # Como são 2 heads com d_out=2, cada token terá 4 valores de context embedding
    mha = MultiHeadAttentionWrapper_v1(d_in,
                                       d_out,
                                       context_length,
                                       0.0,
                                       num_heads=2)
    context_vecs = mha(batch)
    return context_vecs


def compute_all_context_embeddings_v2():
    inputs = _get_input_embeddings()
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 4
    mha = MultiHeadAttention_v2(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    return context_vecs


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
    start = timer()
    result = compute_all_context_embeddings_v1()
    end = timer()
    time_1 = end - start
    print(f"V1 TIME: {time_1:.4f}")
    print(result)
    print("context_vecs.shape:", result.shape)

    start = timer()
    result = compute_all_context_embeddings_v2()
    end = timer()
    time_2 = end - start
    print(f"V2 TIME: {time_2:.4f}")
    print(result)
    print("context_vecs.shape:", result.shape)
    print(f"TIME GAIN: {time_1/time_2:.3f}X")
