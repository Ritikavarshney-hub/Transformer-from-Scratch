import torch 
import torch.nn as nn
import math

class inputembeddings(nn.Module):
    def __init__(self, d_model: int , vocab_size: int):
        super().__init__()
        self.d_model=d_model 
        self.embedding=nn.Embedding(vocab_size, d_model) #embedding layer
        self.vocab_size=vocab_size 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #scale embeddings by sqrt(d_model)


class positionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int ,dropout : float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout) #dropout layer

        pe=torch.zeros(seq_len,d_model) #positional encoding matrix(seq_len, d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #position indices(seq_len, 1)
        div_term=torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) #divergence term
        pe[:,0::2]=torch.sin(position * div_term) #apply sine to even indices
        pe[:,1::2]=torch.cos(position * div_term) #apply cosine to odd indices

        pe=pe.unsqueeze(0) #add batch dimension
        self.register_buffer('pe',pe) #register as buffer to avoid being updated during training

    def forward(self,x):
        x=x + (self.pe[:, :x.shape[1], :]).requires_grad(False) #add positional encoding to input embeddings
        return self.dropout(x) #apply dropout
    
class LayerNormalisation(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-6) -> None:
        super().__init__()
        self.d_model=d_model
        self.eps=eps    
        self.alpha=nn.Parameter(torch.ones(d_model)) #scale parameter
        self.bias=nn.Parameter(torch.zeros(d_model)) #shift parameter

    def forward(self,x):
        mean=x.mean(-1,keepdim=True) #compute mean
        std=x.std(-1,keepdim=True) #compute standard deviation
        return self.alpha * (x - mean) / (std + self.eps) + self.bias #normalize and scale/shift

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff) #first linear layer expands dimensionality from d_model to d_ff
        self.dropout=nn.Dropout(dropout) #dropout layer to prevent overfitting
        self.linear2=nn.Linear(d_ff,d_model) #second linear layer projects back to d_model for residual connection

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) #apply first linear layer, ReLU activation, dropout, and second linear layer to produce output of same shape as input for residual connection
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" #ensure that d_model can be evenly split into num_heads for multi-head attention
        self.d_k=d_model // num_heads #dimensionality of each attention head
        self.q=nn.Linear(d_model,d_model) #linear layer to project input to query space
        self.k=nn.Linear(d_model,d_model) #linear layer to project input to key space
        self.v=nn.Linear(d_model,d_model) #linear layer to project input to value space
        self.out=nn.Linear(d_model,d_model) #linear layer to project concatenated attention outputs back to d_model
        self.dropout=nn.Dropout(dropout) #dropout layer to apply to attention weights for regularization

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_k=query.shape[-1] #dimensionality of each attention head
        scores=query @ key.transpose(-2,-1) / math.sqrt(d_k) #compute scaled dot-product attention scores by multiplying query with transposed key and scaling by sqrt(d_k) to prevent large values that can lead to vanishing gradients
        if mask is not None:
            scores=scores.masked_fill(mask==0, -1e9) #apply mask to attention scores by setting masked positions to a large negative value so that they have negligible influence after softmax
        p_attn=scores.softmax(dim=-1) #apply softmax to attention scores to obtain attention weights that sum to 1 across the key dimension
        if dropout is not None:
            p_attn=dropout(p_attn) #apply dropout to attention weights for regularization during training
        return (p_attn @ value), p_attn #compute weighted sum of value vectors using attention weights to produce output of shape (batch_size, num_heads, seq_len, d_k) and return along with attention weights for visualization

    def forward(self, query, key, value, mask):
        query=self.q(query) #project input to query space using linear layer
        key=self.k(key) #project input to key space using linear layer
        value=self.v(value) #project input to value space using linear layer

        query=query.view(query.shape[1], query.shape[1], self.num_heads, self.d_k).transpose(1,2) #reshape and transpose query to separate heads for multi-head attention, resulting in shape (batch_size, num_heads, seq_len, d_k)
        key=key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2) #reshape and transpose key to separate heads for multi-head attention, resulting in shape (batch_size, num_heads, seq_len, d_k)
        value=value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2) #reshape and transpose value to separate heads for multi-head attention, resulting in shape (batch_size, num_heads, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x=x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.out(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout:float) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalisation(d_model)
        
    def forward(self,x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout) for _ in range(2)])
    
    def forward(self,x, src_mask):
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation(layers[0].self_attention_block.d_model)

    def forward(self,x, src_mask):
        for layer in self.layers:
            x=layer(x, src_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout) for _ in range(3)])
    
    def forward(self,x, memory, src_mask, tgt_mask):
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x, lambda x: self.cross_attention_block(x,memory,memory,src_mask))
        x=self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation(layers[0].self_attention_block.d_model)

    def forward(self,x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x=layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.linear=nn.Linear(d_model, vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.linear(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:inputembeddings, tgt_embed:inputembeddings, pos_encoder:positionalEncoding, pos_decoder:positionalEncoding, projection:ProjectionLayer) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.pos_encoder=pos_encoder
        self.pos_decoder=pos_decoder
        self.projection=projection

    def encode(self, src, src_mask):
        src=self.src_embed(src)
        src=self.pos_encoder(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.pos_decoder(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection(x)
    
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, d_model:int=512, d_ff:int=2048, num_heads:int=8, num_layers:int=6, dropout:float=0.1, seq_len:int=100) -> Transformer:
    #create embedding layers
    src_embed=inputembeddings(d_model, src_vocab_size)
    tgt_embed=inputembeddings(d_model, tgt_vocab_size)
    #create positional encoding layers
    pos_encoder=positionalEncoding(d_model, seq_len, dropout)
    pos_decoder=positionalEncoding(d_model, seq_len, dropout)
    
    encoder_blocks=[]
    for _ in range(num_layers):
        encoder_self_attention_block=MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block=EncoderLayer(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(num_layers):
        decoder_self_attention_block=MultiHeadAttention(d_model, num_heads, dropout)
        cross_attention_block=MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block=DecoderLayer(decoder_self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection=ProjectionLayer(d_model, tgt_vocab_size)

    Transformer_model=Transformer(encoder, decoder, src_embed, tgt_embed, pos_encoder, pos_decoder, projection)

    for p in Transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return Transformer_model


        