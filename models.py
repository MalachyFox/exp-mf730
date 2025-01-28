import json
import torch
import torch.nn as nn
from hyperparams import HyperParams
    
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = hps.bidirectional
        self.hidden_size = hps.hidden_size
        self.batch_size = hps.batch_size
        num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=hps.input_size,
            hidden_size=hps.hidden_size,
            num_layers=hps.num_layers,
            batch_first=True,
            bidirectional= bool(hps.bidirectional),
            dropout=hps.dropout
        )

        self.linear1 = nn.Linear(in_features=hps.hidden_size * num_directions, out_features=hps.linear_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hps.linear_size, out_features=1)
        self.dropout = nn.Dropout(hps.dropout)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            out = torch.cat((cell[-2], cell[-1]), dim=1).reshape(self.batch_size,2 * self.hidden_size)
        else:
            out = cell[-1].reshape(self.batch_size,self.hidden_size)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

    def loss_func(self):
        return nn.BCEWithLogitsLoss()
