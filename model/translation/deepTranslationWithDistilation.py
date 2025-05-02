import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dims, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) *
                             (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(max_len, 1, embedding_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayerManual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayerManual, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src,  src_key_padding_mask=None):
        # Self-attention block
        attn_output, _ = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout(attn_output))

        # Feedforward block
        ff_output = self.linear2(self.dropout_ff(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff_output))

        return src


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayerManual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerDecoderLayerManual, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, src, tgt_mask=None,
                tgt_key_padding_mask=None, src_key_padding_mask=None):
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2, _ = self.multihead_attn(
            tgt, src, src,
            key_padding_mask=src_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        ff = self.dropout_ff(F.relu(self.linear1(tgt)))
        tgt2 = self.linear2(ff)
        tgt = self.norm3(tgt + self.dropout2(tgt2))

        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_dims ** -0.5)
        self.embedding_dims = embedding_dims
        self.pos = PositionalEncoding(embedding_dims, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerManual(
                d_model=embedding_dims,
                nhead=n_heads,
                dim_feedforward=hidden_dims,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_key_padding_masks=None):
        # Embedding
        src = self.embedding(src) * math.sqrt(self.embedding_dims)

        
        src = self.pos(src)

        
        for i, layer in enumerate(self.encoder_layers):
            src = layer(src, src_key_padding_mask=src_key_padding_masks)

        
        src = self.dropout(src)

        return src

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout=0.5):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_dims ** -0.5)
        self.embedding_dims = embedding_dims
        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerManual(
                d_model=embedding_dims,
                nhead=n_heads,
                dim_feedforward=hidden_dims,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        self.linear = nn.Linear(embedding_dims, vocab_size)

    def forward(self, tgt, src, tgt_key_padding_mask=None, src_key_padding_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_dims)
        tgt = self.pos_encoder(tgt)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
  
        
        for i, layer in enumerate(self.decoder_layers):
            tgt = layer(tgt, src, tgt_mask=tgt_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       src_key_padding_mask=src_key_padding_mask)

        
        output = self.linear(tgt)

        
        return output

class TransformerSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout=0.5):
        super(TransformerSeq2SeqModel, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout)
        self.decoder = TransformerDecoder(vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout)

    def forward(self,src,tgt, src_key_padding_mask=None , tgt_key_padding_mask=None):
        src = self.encoder(src,src_key_padding_masks= src_key_padding_mask)
        output = self.decoder(tgt,src,tgt_key_padding_mask=tgt_key_padding_mask,
                        src_key_padding_mask=src_key_padding_mask)
        return output