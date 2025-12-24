import torch
from torch.utils.data import Dataset, DataLoader


class LongSequenceDataset(Dataset):
    def __init__(self, tokens, seq_len):
        """
        tokens: 1D LongTensor of shape (N,)
        seq_len: length of input sequence
        """
        assert tokens.dim() == 1
        self.tokens = tokens
        self.seq_len = seq_len

        # last possible start index is N - seq_len - 1 (need one extra token for target)
        self.max_start = len(tokens) - seq_len - 1
        assert self.max_start > 0, "Sequence too short for given seq_len"

    def __len__(self):
        return self.max_start + 1

    def __getitem__(self, idx):
        # input: tokens[idx : idx+seq_len]
        # target: next token after the window
        x = self.tokens[idx: idx + self.seq_len]  # (seq_len,)
        y = self.tokens[idx + self.seq_len]  # scalar (next int)
        return x.long(), y.long()


class MultiSequenceDataset(Dataset):
    def __init__(self, sequences, seq_len):
        """
        sequences: list of 1D LongTensors or lists of ints
        """
        self.seq_len = seq_len
        self.samples = []  # (seq_id, start_idx)

        # standardize to tensors
        self.sequences = [torch.as_tensor(s, dtype=torch.long) for s in sequences]

        for seq_id, seq in enumerate(self.sequences):
            max_start = len(seq) - seq_len - 1
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                self.samples.append((seq_id, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, start = self.samples[idx]
        seq = self.sequences[seq_id]
        x = seq[start: start + self.seq_len]
        y = seq[start + self.seq_len]
        return x.long(), y.long()
