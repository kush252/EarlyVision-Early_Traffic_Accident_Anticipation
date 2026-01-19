import torch
import torch.nn as nn

class RiskLSTM(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        hidden_dim=256,
        num_layers=1,
        bidirectional=False
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden_dim * direction_factor, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        """

        lstm_out, (h_n, c_n) = self.lstm(x)

        # h_n shape:
        # (num_layers * directions, batch, hidden_dim)

        if self.lstm.bidirectional:
            # last layer forward & backward
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.fc(last_hidden)

        return logits
