import urllib.request
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt,
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
    """
    Args:
        batch_size (int):
            O tamanho de cada batch de sentenças
        max_length (int):
            Quantos tokens formarão uma sentença
        stride (int):
            O tamanho da janela deslizante para criar os pares
        shuffle (bool):
            Indica se os batches virão aleatorizados (True) ou não (False)
        drop_last (bool):
            Se devemos desconsiderar o último batch para evitar que ele tenha
            um tamanho menor do que o esperado
        num_workers (int):
            Quantidade de threads a serem usadas para carregar os batches. 0 é tudo
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # 1. Conseguindo um texto
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Esse é o tokenizer já implementado
    ## O tokenizer do gpt2 e do gpt3 é o Byte-Pair-Encoding (BPE)
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    # 3. Criando pares para serem preditos pela LLM
    max_length = 4
    dataloader = create_dataloader_v1(raw_text,
                                      batch_size=8,
                                      max_length=max_length,
                                      stride=max_length,
                                      shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

    # 4. Criando embeddings para os tokens
    ## O tamanho do vocabulário do tiktoken do gpt2 é 50257
    ## Da mesma forma, podemos querer representar cada token com 256 dimensões
    vocab_size = 50257
    output_dim = 256
    ## Essa 'camada' vai agir simplesmente como uma lookup table ao informar o id do token
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    ## Como os inputs são ids de tokens, eles acessam tensores específicos dentro da camada de representação
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    ## 4.1 Além do embedding 'normal' também temos o embedding posicional
    ## Nesse caso, o embedding de cada posição tem 256 dimensões também
    ## Como estamos lidando com sequências de tamanho 4 (context_length)
    ## e estamos usando absolute-positional-encoding, temos 4 embeddings de 256 dims no total.
    ## A OpenAI usa de absolute-positional-encoding que é otimizado ao longo do treinamento
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    ## 4.2 Aqui, adicionamos a informação posicional aos embeddings dos tokens
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    # O embedding inicial é diferente do embedding final já que estamos adicionando o pos-embedding
    print(token_embeddings[0][0])
    print(input_embeddings[0][0])
