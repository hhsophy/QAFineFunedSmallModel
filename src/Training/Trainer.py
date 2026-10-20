import torch
from src.evaluation import calc_loss_loader,calc_loss_batch,evaluate_model

from .utils import generate_and_print_simple

def train_model(model,train_loader,val_loader,
                       optimizer,device, num_epochs,
                       eval_freq,eval_iter,start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []   
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()

            tokens_seen +=input_batch.numel()
            global_step +=1

            if global_step % eval_freq==0:
                train_loss, val_loss= evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, "f"Val loss {val_loss:.3f}")

        generate_and_print_simple(model,tokenizer,device,start_context)

    return train_losses,val_losses,track_tokens_seen



def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################
    path="D:\\codes\\text.txt"
    with open(path, "r", encoding="utf-8") as f:
        text_data = f.read()
    text_data=re.sub(r'\d+'," ", raw_text) # remove numbers from the text 
    ##############################
    # Initialize model
    ##############################

    model = LLMModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )


    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_data_loader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_data_loader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="enemy forces", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model