import tiktoken
from timeit import default_timer as timer
import torch

from src.transformer_block.config import GPT_CONFIG_124M
from src.transformer_block.real_gpt_model_imp import GPTModel


def get_model():
    start = timer()
    model = GPTModel(GPT_CONFIG_124M)
    end = timer()
    print(f"Modelo definido em {end-start:.4f} segundos")
    return model


def process_two_sentences(model):
    print("Conseguindo o tokenizador")
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    print("Tokenizando inputs")
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    print("Executando modelo...")
    start = timer()
    out = model(batch)
    end = timer()
    print(f"Modelo executado em {end-start:.4f} segundos")
    print("Input shape:", batch.shape)
    print("Output shape:", out.shape)


def complete_text(model):
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    # O unsqueeze adiciona a dimensão de batch
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval()
    start = timer()
    out = generate_text_simple(model=model,
                               idx=encoded_tensor,
                               max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    end = timer()
    print(f"Modelo executado em {end-start:.4f} segundos")
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Implementa um loop generativo simples
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # Transforma os logits em probabilidades
        # Tecnicamente, como o softmax é monotônico
        # não precisamos dele, mas ele ainda assim é usado
        probas = torch.softmax(logits, dim=-1)
        # Pega o id do token mais provável (greedy-decoding)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    torch.manual_seed(123)
    model = get_model()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    process_two_sentences(model)
    complete_text(model)
