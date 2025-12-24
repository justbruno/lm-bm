import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class NextIntTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Use last position representation to predict next token
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len) integer token indices
        returns: logits (batch, vocab_size)
        """
        emb = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)
        enc_out = self.encoder(emb)  # (batch, seq_len, d_model)
        last = enc_out[:, -1, :]  # (batch, d_model)
        logits = self.fc_out(last)  # (batch, vocab_size)
        return logits
