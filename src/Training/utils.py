
import torch
import tiktoken
import json
from pathlib import Path
tokenizer=tiktoken.get_encoding("gpt2")

# convert from text to token IDs
def text_to_token_ids(text,tokenizer=None):
    if tokenizer is None:
        tokenizer=tokenizer
    
    encoded=tokenizer.encode(text, allowed_special={"|<endoftext>|"})
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# Reverse Back the token IDs to thier corresponding words
def token_ids_to_text(token_ids,tokenizer):
    if tokenizer is None:
        tokenizer=tokenizer
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# generate the words based of the logits output of the model with topk,temp,and multinomial stratiges
def generate(model,idx,max_new_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)

        logits=logits[:,-1,:]
        if top_k is not None:
            top_logits,_=torch.topk(logits,top_k)
            min_val=top_logits[:,-1]
            logits=torch.where(logits<min_val, torch.tensor(float('-inf')).to(top_logits.device),logits)
        if temperature > 0.0:
            logits=logits/temperature
            probas=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probas,num_samples=1)
        else:
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        
        idx=torch.cat((idx,idx_next), dim=1)

    return idx

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                   
        probas = torch.softmax(logits, dim=-1)          
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   
        idx = torch.cat((idx, idx_next), dim=1)    
    
    return idx

def generate_and_print_simple(model,tokenizer,device,start_context):
    model.eval()
    contex_size=model.pos_emb.weight.shape[0]
    encoded=text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(model=model,idx=encoded,max_new_tokens=50,context_size=contex_size)

    decoded_text=token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n", " "))     
    model.train()

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_losses(train_losses=[], test_losses=[], val_losses=[],
                path="results/losses.json", resume=True):

    path = Path(path)

    if resume and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            losses = json.load(f)
    else:
        losses = {
            "train": [],
            "test": [],
            "val": []
        }

    # Make sure all three keys exist
    for key in ("train", "test", "val"):
        losses.setdefault(key, [])

    # Append new losses
    losses["train"].extend(train_losses)
    losses["test"].extend(test_losses)
    losses["val"].extend(val_losses)

    # Create the directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create the file or overwrite the existing file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=4)

