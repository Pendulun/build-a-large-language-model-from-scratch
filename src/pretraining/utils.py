import torch
from src.transformer_block._2_define_real_gpt_like_model import generate_text_simple
from src.tokenizer.tokenizer import create_dataloader_v1


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def load_text_data() -> str:
    file_path = "src/tokenizer/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def get_train_val_dataloaders(text_data: str, config: dict):
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(train_data,
                                        batch_size=2,
                                        max_length=config["context_length"],
                                        stride=config["context_length"],
                                        drop_last=True,
                                        shuffle=True,
                                        num_workers=0)

    val_loader = create_dataloader_v1(val_data,
                                      batch_size=2,
                                      max_length=config["context_length"],
                                      stride=config["context_length"],
                                      drop_last=False,
                                      shuffle=False,
                                      num_workers=0)

    return train_loader, val_loader


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                      val_loader, device,
                                                      eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1),
                                             target_batch.flatten())
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,
                                      model,
                                      device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,
                                    model,
                                    device,
                                    num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,
                                         idx=encoded,
                                         max_new_tokens=50,
                                         context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
