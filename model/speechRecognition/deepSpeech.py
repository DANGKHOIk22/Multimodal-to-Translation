import math
import torch
import torch.nn as nn
import timm

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


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dims,
            n_heads,
            hidden_dims,
            dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)


    def forward(self, src):
        src = self.transformer_encoder(src)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dims, n_heads, hidden_dims, n_layers, dropout=0.5):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_dims ** -0.5)
        self.embedding_dims = embedding_dims
        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dims,
            nhead=n_heads,
            dim_feedforward=hidden_dims,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        # self.linear = nn.Linear(embedding_dims, vocab_size)

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_dims)
        tgt = self.pos_encoder(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        tgt = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # tgt = self.linear(tgt)

        return tgt


class DeepSpeech(nn.Module):
    def __init__(self, vocab_size, hidden_size=256,n_heads = 8, n_layers=2, dropout=0.2,unfreeze_layers= 2):
        super(DeepSpeech, self).__init__()

        # Load pretrained ResNet34 with grayscale input
        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)

        # ↓↓↓ MODIFY LAYER4 ↓↓↓
        # Change layer4's stride to (1, 1) to avoid 2x downsampling
        backbone.layer3[0].conv1.stride = (1, 1)
        backbone.layer3[0].downsample[0].stride = (1, 1)
        backbone.layer4[0].conv1.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)


        # Remove final layers and add custom pool
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((None, 1)))
        self.backbone = nn.Sequential(*modules)

        # Freeze early layers, fine-tune last two blocks
        for parameter in self.backbone[:-unfreeze_layers].parameters():
            parameter.requires_grad = False
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.map_fc = None
        self.input_dim = None
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = TransformerEncoder(vocab_size, hidden_size, n_heads, hidden_size // 2, n_layers, dropout)
        self.decoder = TransformerDecoder(vocab_size, hidden_size, n_heads, hidden_size // 2, n_layers, dropout)
        self.classifier = nn.Sequential()
        self.vocab_size = vocab_size

    def forward(self, x,tgt, tgt_key_padding_mask=None):
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.backbone(x)  # (B, C, T//8, 1)

        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)  # (B, T, C)

        if self.map_fc is None:
            self.input_dim = C * F
            self.map_fc = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU()
            ).to(x.device)
            self.project_encode = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size , self.vocab_size - 2),
                nn.LogSoftmax(dim=2)
               ).to(x.device)
            self.project_decode = nn.Sequential(
                nn.Linear(self.hidden_size , self.vocab_size)
               ).to(x.device)

        mapped_features = self.map_fc(x)
        encoder_output = self.encoder(mapped_features)
        decoder_output = self.decoder(tgt, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask)
    
        encoder_output = self.project_encode(encoder_output)
        decoder_output = self.project_decode(decoder_output)
    
        return encoder_output.permute(1, 0, 2), decoder_output.permute(0, 2, 1)  # (T, B, C), (B, C, L)