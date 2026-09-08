import torch.nn as nn

class DenoiseGRU(nn.Module):
    def __init__(self, freq_bins=257, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        # Double the hidden size for the linear layer if bidirectional
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, freq_bins)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)
