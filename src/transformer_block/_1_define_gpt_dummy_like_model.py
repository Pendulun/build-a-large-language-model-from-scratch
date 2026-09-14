import tiktoken
from timeit import default_timer as timer
import torch

from src.transformer_block.config import GPT_CONFIG_124M
from src.transformer_block.dummy_gpt_model import DummyGPTModel

if __name__ == "__main__":
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

    torch.manual_seed(123)
    print("Inicializando modelo gpt-2...")
    start = timer()
    model = DummyGPTModel(GPT_CONFIG_124M)
    end = timer()
    print(f"Modelo inicializado em {end-start:.4f} segundos")
    print("Processando batch de shape", batch.shape)
    start = timer()
    logits = model(batch)
    end = timer()
    print(f"Processado em {end-start:.4f} segundos")
    print("Output shape:", logits.shape)
    print(logits)
