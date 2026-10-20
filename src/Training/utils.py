
import torch
import tiktoken
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