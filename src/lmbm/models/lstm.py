import torch
import torch.nn as nn

class NextIntLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                           dropout=dropout)  # (batch, seq, feat)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) of integer indices
        x = self.embedding(x)               # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden)  # out: (batch, seq_len, hidden_dim)
        # Use only the last time step for next-token prediction
        last_hidden = out[:, -1, :]         # (batch, hidden_dim)
        logits = self.fc(last_hidden)       # (batch, vocab_size)
        return logits
