import torch


def compute_one_context_vector(target_word_id: int = 1):
    inputs = _get_input_embeddings()
    query = inputs[target_word_id]
    print("Embedding inicial da segunda palavra da sentença:", query)

    # Calculamos o score de atenção de toda palavra na sentença com a palavra alvo
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        # Quando x_i == query, isso atinge o máximo.
        attn_scores_2[i] = torch.dot(x_i, query)

    print(attn_scores_2)

    # Normalizamos os scores para que a soma seja 1.
    # Esse é um jeito
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # O jeito mais recomendado é usar uma softmax para normalizar
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights SOFTMAX:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    # A implementação direta do torch é mais otimizada
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights SOFTMAX PYTORCH:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # Agora, o vetor de contexto para a segunda palavra da sentença
    # é uma soma ponderada dos vetores de entrada pelos scores de atenção
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print("Vetor de contexto final:", context_vec_2)
    return context_vec_2


def compute_all_context_vectors():
    # O input é uma matriz n X d em que n é a quantidade de palavras e d é o
    # tamanho do embedding de cada palavra
    inputs = _get_input_embeddings()
    print("Embeddings dos tokens da sentença:", inputs, sep="\n")

    # Os scores de atenção são uma matriz n X n onde n é o tamanho da sentença
    attn_scores = inputs @ inputs.T
    print("Attention scores:", attn_scores, sep="\n")

    # Uma vez normalizados usando o softmax, se tornam 'pesos de atenção'.
    # O softmax está sendo aplicado sobre a última dimensão do tensor (-1)
    # no caso, as colunas. Isso significa que a operação será aplicada sobre os valores
    # ao longo do eixo da coluna e não sobre o eixo das linhas
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("Attention weights:", attn_weights, sep="\n")

    # Agora aplicamos os pesos sobre as entradas para computar o embedding de contexto
    # Uma multiplicação (n,n) X (n,d) = (n,d)
    all_context_vecs = attn_weights @ inputs
    print("Context embeddings:", all_context_vecs, sep="\n")


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


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


if __name__ == "__main__":
    compute_all_context_vectors()
