import tiktoken
import torch
from src.transformer_block.real_gpt_model_imp import GPTModel
from src.transformer_block.config import GPT_CONFIG_124M
from src.pretraining.utils import train_model_simple, get_train_val_dataloaders, load_text_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and optimizer state...")
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    config = dict(GPT_CONFIG_124M)
    # Para permitir treinamento local, diminuímos o tamanho da janela de contexto
    config['context_length'] = 256
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=5e-4,
                                  weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Done!")

    model.train()
    print("Model on train mode")
    print("Training for one more epoch...")
    train_data = load_text_data()
    train_loader, val_loader = get_train_val_dataloaders(train_data, config)
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 1
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
