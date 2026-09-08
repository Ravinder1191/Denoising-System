import json
import torch
import torch.nn as nn
from Config.paths import device, best_params_path, best_model

class DenoiseGRU_IRM(nn.Module):
    def __init__(self, freq_bins=257, hidden_size=224, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out))

def load_trained_model():
    params_path = best_params_path
    weights_path = best_model

    with open(params_path, "r") as f:
        params = json.load(f)

    model = DenoiseGRU_IRM(
        freq_bins=params.get("freq_bins", 257),
        hidden_size=params.get("hidden_size", 224),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.2),
        bidirectional=params.get("bidirectional", False)
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("IRM Model loaded successfully.")
    return model, params
