import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F


class SDPA(nn.Module):
    def __init__(self, input_dim,Output_dim,context_length,dropout,num_head,qkv_bias=False):
        super().__init__()
        assert(Output_dim % num_head == 0) , "output_dim must be divisable by number of head"

        self.Output_dim=Output_dim
        self.num_heads=num_head
        self.heads_dim= Output_dim // num_head
        self.W_q=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.W_k=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.W_v=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.out_porj=nn.Linear(self.Output_dim,self.Output_dim)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        batch_size, seq_len, emb_dim=x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        keys=K.view(batch_size, seq_len,self.num_heads, self.heads_dim).transpose(1,2)
        values=V.view(batch_size, seq_len,self.num_heads, self.heads_dim).transpose(1,2)
        queries=Q.view(batch_size, seq_len,self.num_heads, self.heads_dim).transpose(1,2)
        
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # 4. Merge heads
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, seq_len, emb_dim)

        # 5. Final output projection
        context_vec = self.out_porj(context_vec)

        return context_vec


class MultiHeadAtention(nn.Module):
    def __init__(self, input_dim,Output_dim,context_length,dropout,num_head,qkv_bias=False):
        super().__init__()
        assert(Output_dim % num_head == 0) , "output_dim must be divisable by number of head"

        self.Output_dim=Output_dim
        self.num_heads=num_head
        self.heads_dim= Output_dim // num_head
        self.W_q=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.W_k=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.W_v=nn.Linear(input_dim,Output_dim,bias=qkv_bias)
        self.out_porj=nn.Linear(self.Output_dim,self.Output_dim)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_token, dim_in=x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        keys=K.view(b, num_token,self.num_heads, self.heads_dim)
        values=V.view(b, num_token,self.num_heads, self.heads_dim)
        queries=Q.view(b, num_token,self.num_heads, self.heads_dim)

        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        queries=queries.transpose(1,2)

        atten_score=queries @ keys.transpose(2,3)
        mask_bool=self.mask.bool()[:num_token, :num_token]
        atten_score.masked_fill_(mask_bool,-torch.inf)
        atten_weights=torch.softmax(atten_score/keys.shape[-1]**0.5, dim=-1)
        atten_weights=self.dropout(atten_weights)
        context_vec=(atten_weights @ values).transpose(1,2)
        context_vec=context_vec.contiguous().view(b, num_token, self.Output_dim)
        context_vec=self.out_porj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(embed_dim))
        self.shift=nn.Parameter(torch.zeros(embed_dim))
        self.eps=1e-5

    def forward(self,x):
        x_mean=x.mean(dim=-1,keepdim=True)
        x_var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-x_mean)/torch.sqrt(x_var+self.eps)

        return self.scale*norm_x+self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))*(x + 0.044715 * torch.pow(x,3))))


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]), 
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]))

    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=SDPA(input_dim=cfg['emb_dim'],Output_dim=cfg['emb_dim'],
                                    context_length=cfg['context_length'],dropout=cfg['drop_rate'],
                                    num_head=cfg['n_heads'],qkv_bias=cfg['qkv_bias'])
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg['emb_dim'])
        self.norm2=LayerNorm(cfg['emb_dim'])
        self.drop_shortcut=nn.Dropout(cfg['drop_rate'])


    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
            
            
        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=x+shortcut

        return x

class LLMModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])
        self.trf_blocks=nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)


    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits
