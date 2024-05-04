import torch 

import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create positional embeddings
        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=float).unsqueeze(1)  # [seq_len, 1]

        # [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log2(10000) / d_model)) # e^((2i/d_model) * log (1/10000)) = (10000)^(-2i/d_model)

        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        pe = pe.unsqueeze(0) # [1, seq_len, d_model] this is form batch processing

        self.register_buffer('pe', pe)  
        # If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.

    def forward(self, x):
        # x  = [batch_size, len, d_model]
        x = x + self.pe[:, 0:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model)) # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # added

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True)  # [batch_size, len, 1]  #keepdim keep last dim as 1
        std = x.std(dim = -1, keepdim=True) # [batch_size, len, 1]
        return self.gamma * ((x - mean)/ (std + self.eps)) + self.bias
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # [d_model * d_ff]
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # [d_ff, d_model]

    def forward(self, x):
        # x = [batch_size, len, d_model]
        x = self.linear_1(x) # [batch_size, len, d_ff]
        x  = torch.relu(x)
        x =  self.linear_2(self.dropout(x)) # [batch_size, len, d_model]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h ==0, "d_model is not divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod   #this allows us to call this function without instatiating an object of this class
    def attention(q,k,v,mask, dropout: nn.Dropout):
        # q, k, v  ->  [batch_size, h, len, d_k] 
        d_k = q.shape[-1]

        # [batch_size, h, len, d_k]  @ [batch_size, h, d_k, len] -> [batch_size, h, len, len]
        attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(d_k) # [batch_size, h, len, len]

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) # [batch_size, h, len, len]

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # [batch_size, h, len, len] @ [batch_size, h, len, d_k] ->  [batch_size, h, len, d_k] 
        return (attention_scores @ v), attention_scores


    def forward(self, q, k, v, mask):
        q_ = self.w_q(q)  # [batch_size, len, d_model] -> [batch_size, len, d_model]
        k_ = self.w_k(k)  # [batch_size, len, d_model] -> [batch_size, len, d_model]
        v_ = self.w_v(v)  # [batch_size, len, d_model] -> [batch_size, len, d_model]

        # split it into heads [batch_size, h, len, d_k]  

         # [batch_size, len, d_model]  -view->  [batch_size,len, h, d_k]   -Transpose->  [batch_size, h, len, d_k] 
        q_heads = q_.view(q_.shape[0],q_.shape[1], self.h, self.d_k).transpose(1,2) 
        k_heads = k_.view(k_.shape[0],k_.shape[1], self.h, self.d_k).transpose(1,2)
        v_heads = v_.view(v_.shape[0],v_.shape[1], self.h, self.d_k).transpose(1,2)  

    

        # calculate attention scores
        x, self.attention_scores = MultiHeadAttention.attention(q_heads, k_heads, v_heads, mask, self.dropout)

        # concatenate the heads now
        x = x.transpose(1,2).contiguous() #  [batch_size, h, len, d_k] -> [batch_size,len, h, d_k] 
        x = x.view(x.shape[0], x.shape[1], self.d_model) #[batch_size, len , d_model]

        return self.w_o(x) # [batch_size, len , d_model]
    
class AdditionLayer(nn.Module):
    def __init__(self, d_model,dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, prev_layer):    
        return x + self.dropout(prev_layer(self.norm(x)))   # ( skipper connections )
    
class Encoder(nn.Module):
    def __init__(self,d_model:int, self_attention: MultiHeadAttention, feed_forward_net: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward_net
        self.connections = nn.ModuleList([AdditionLayer(d_model,dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.connections[0](x, lambda x: self.self_attention(x,x,x, src_mask))
        x = self.connections[1](x, self.feed_forward)
        return x
    
class EncoderStack(nn.Module):
    def __init__(self, d_model:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model:int,self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward_net: FeedForward, dropout: float ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward_net
        self.connections = nn.ModuleList([AdditionLayer(d_model, dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.connections[0](x, lambda x: self.self_attention(x, x,x, tgt_mask))
        x = self.connections[1](x, lambda x: self.cross_attention(x,encoder_output, encoder_output, src_mask))
        x = self.connections[2](x, self.feed_forward)
        return self.dropout(x)
    
class DecoderStack(nn.Module):
    def __init__(self, d_model:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask ):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    

class VocabProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size:int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size) # x = [batch_size, len, d_model] -> [batch_size, len, vocab_size]

    def forward(self, x):
        # x = [batch_size, len, d_model]
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: EncoderStack, decoder: DecoderStack, src_embeddings: InputEmbeddings, tgt_embeddings: InputEmbeddings , src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: VocabProjectionLayer) -> None:
        super().__init__()
        self.encoder =  encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, x, src_mask):
        x = self.src_embeddings(x)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x
    
    def decode(self,x ,encoder_output, src_mask, tgt_mask):
        x = self.tgt_embeddings(x)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x
    
    def project(self, x):
        return self.projection(x)
    


def build_transformer(src_vocab_size:int, out_vocab_size:int, src_max_len:int, out_max_len:int, d_model:int = 512, N: int = 6, h:int = 8, dropout: float = 0.1, d_ff:int = 2048) -> Transformer:

    src_emb = InputEmbeddings(d_model, src_vocab_size)
    out_emb = InputEmbeddings(d_model, out_vocab_size)

    src_pe = PositionalEncoding(d_model, src_max_len, dropout)
    out_pe = PositionalEncoding(d_model, out_max_len, dropout)

    encoder_list = []
    for _ in range(N):
        mha = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        encoder_list.append(Encoder(d_model, mha, ffn, dropout))

    encoder = EncoderStack(d_model, nn.ModuleList(encoder_list))

    decoder_list = []
    for _ in range(N):
         mha = MultiHeadAttention(d_model, h, dropout)
         cmha = MultiHeadAttention(d_model, h, dropout)
         ffn = FeedForward(d_model, d_ff, dropout)
         decoder_list.append(Decoder(d_model, mha, cmha, ffn, dropout))
    
    decoder = DecoderStack(d_model, nn.ModuleList(decoder_list))

    projectionLayer = VocabProjectionLayer(d_model, out_vocab_size)


    transformer = Transformer(encoder, decoder, src_emb, out_emb, src_pe, out_pe, projectionLayer)


    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer









    


        


       

    

    

        
