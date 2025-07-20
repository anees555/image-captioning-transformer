import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=8, num_layers=3, dropout=0.1, max_len=50):
        super(ImageCaptioningModel, self).__init__()

        # Encoder: CNN (ResNet50)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove the last fc & pooling
        self.encoder_cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # to get consistent spatial size

        # Project ResNet features to embedding dim for decoder
        self.enc_to_embed = nn.Linear(2048, embed_dim)

        # Decoder: Transformer Decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, captions):
        """
        images: [batch, 3, 224, 224]
        captions: [batch, seq_len]
        """
        batch_size = images.size(0)

        # ==============
        # Encoder
        # ==============
        features = self.encoder_cnn(images)  # [batch, 2048, H/32, W/32]
        features = self.adaptive_pool(features)  # [batch, 2048, 14, 14]
        features = features.flatten(2).permute(2, 0, 1)  # [14*14, batch, 2048]
        features = self.enc_to_embed(features)  # [14*14, batch, embed_dim]

        # Decoder
  
        # Embed captions
        captions_emb = self.embedding(captions).permute(1, 0, 2)  # [seq_len, batch, embed_dim]
        captions_emb = self.pos_encoder(captions_emb)

        tgt_mask = self.generate_square_subsequent_mask(captions_emb.size(0)).to(captions.device)

        out = self.transformer_decoder(
            tgt=captions_emb,
            memory=features,
            tgt_mask=tgt_mask
        )  # [seq_len, batch, embed_dim]

        out = self.fc_out(out)  # [seq_len, batch, vocab_size]
        return out.permute(1, 0, 2)  # [batch, seq_len, vocab_size]

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        if d_model % 2 == 1:  # Handle odd dimensions
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # dim 2i+1, exclude last element
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
            
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


