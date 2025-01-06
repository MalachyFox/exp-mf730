import json
import torch
import torch.nn as nn
from hyperparams import HyperParams

class LSTMClassifier(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        return out