import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (T, 1, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, D)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LandmarkTransformer(nn.Module):
    def __init__(self, num_classes=26, input_dim=63, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super(LandmarkTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: (Batch, Frames, InputDim)
        """
        x = x.permute(1, 0, 2) # (Frames, Batch, InputDim) needed for Transformer
        
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        
        # Global Average Pooling over time? Or take last token?
        # For classification, taking the mean over time is often robust.
        # Alternatively, take the [CLS] token equivalent if we had prepended one.
        # Let's use Mean Pooling manually.
        output = output.mean(dim=0) # (Batch, d_model)
        
        logits = self.classifier(output)
        return logits
