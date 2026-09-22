import tiktoken
import torch
from src.transformer_block.real_gpt_model_imp import GPTModel
from src.transformer_block.config import GPT_CONFIG_124M
from src.transformer_block._2_define_real_gpt_like_model import generate_text_simple
from src.pretraining.utils import text_to_token_ids, token_ids_to_text, train_model_simple, get_train_val_dataloaders, load_text_data


def generate(model,
             idx,
             max_new_tokens,
             context_size,
             temperature=0.0,
             top_k=None,
             eos_id=None):
    """
    Método que usa de probabilistic sampling, top-k e temperature scaling para
    gerar tokens de saída
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float('-inf')).to(logits.device),
                                 logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    config = dict(GPT_CONFIG_124M)
    # Para permitir treinamento local, diminuímos o tamanho da janela de contexto
    config['context_length'] = 256

    torch.manual_seed(123)
    model = GPTModel(config)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"])
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    train_data = load_text_data()
    total_characters = len(train_data)
    total_tokens = len(tokenizer.encode(train_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    train_loader, val_loader = get_train_val_dataloaders(train_data, config)
    print("Train loader len:", len(train_loader))
    print("Validation loader len:", len(val_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    model = GPTModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.0004,
                                  weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer)

    # Colocamos o modelo de volta na cpu e no modo avaliação para
    # desligar o dropout
    model.to("cpu")
    model.eval()

    # Geramos um texto na forma greedy
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"])
    print("Greedy Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # Geramos um texto usando de temperatura, top-k e amostragem probabilística
    torch.manual_seed(123)
    token_ids = generate(model=model,
                         idx=text_to_token_ids("Every effort moves you",
                                               tokenizer),
                         max_new_tokens=15,
                         context_size=GPT_CONFIG_124M["context_length"],
                         top_k=25,
                         temperature=1.4)
    print("Probabilistic Output text:\n",
          token_ids_to_text(token_ids, tokenizer))

    target_file = "model_and_optimizer.pth"
    print("Saving model and optimizer state to", )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, target_file)
    print("Saved!")
