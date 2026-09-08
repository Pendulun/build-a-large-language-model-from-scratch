import torch
from src.self_attention._1_1_self_attention_class import SelfAttention_v1


def compute_one_context_vector(target_token_id: int = 1):
    """
    Diferentemente da versão simplificada em que usamos sempre o mesmo vetor
    para computar o context embedding, agora usamos 3 vetores diferentes:
    W_Q, W_K, W_V.
    O W_Q se aproxima mais ao vetor original, sendo usado como base para as contas
    O W_K é usado em conjunto com o W_Q para computar o Attention Score
    Já o W_V é usado em conjunto com o attention weight para gerar o context embedding
    """
    # O inputs é uma matriz n X d com n a quantidade de tokens na sentença
    inputs = _get_input_embeddings()
    print("Inputs:", inputs, sep="\n")

    # x_2 é um vetor 1 x d
    x_2 = inputs[target_token_id]
    print("Initial target token embedding:", x_2, sep="\n")
    # Normalmente, d_in e d_out são iguais mas vamos definir
    # como diferentes para facilitar
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    # Se fôssemos treinar esses parâmetros agora, deixaríamos o requires_grad como True
    # Aqui, simulamos ter as matrizes aprendidas.
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    print("W_Q shape:", W_query.shape)
    print("W_K shape:", W_key.shape)
    print("W_V shape:", W_value.shape)

    # Query é um vetor 1 x d_out
    query_2 = x_2 @ W_query
    print("Query for target token:", query_2, sep="\n")

    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    # Conseguindo os scores
    attn_scores_2 = query_2 @ keys.T
    print("Attention Scores:", attn_scores_2, sep='\n')

    # Conseguindo os attention weights
    d_k = keys.shape[-1]
    # Essa 'escala' pela raiz da dimensionalidade é parte do motivo pelo qual
    # os transformers do GPT usam o chamado 'scaled-dot product attention'.
    # Basicamente, isso ajuda durante o treinamento ao não gerar gradientes pequenos
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print("Attention Scores:", attn_weights_2, sep='\n')

    # Conseguindo o context embedding alvo
    context_vec_2 = attn_weights_2 @ values
    print("Context embedding:", context_vec_2, sep='\n')


def compute_all_context_embeddings():
    inputs = _get_input_embeddings()
    torch.manual_seed(123)
    d_in = inputs.shape[1]
    d_out = 2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))


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
    compute_all_context_embeddings()
