import torch
import torch.nn as nn

class NextIntMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len,
                 hidden_dim=256, num_hidden_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        input_dim = seq_len * embed_dim

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Optional extra hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer: logits over vocabulary
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len) of token indices
        emb = self.embedding(x)                    # (batch, seq_len, embed_dim)
        flat = emb.view(emb.size(0), -1)           # (batch, seq_len*embed_dim)
        logits = self.mlp(flat)                    # (batch, vocab_size)
        return logits
