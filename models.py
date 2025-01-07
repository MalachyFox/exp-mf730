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
    
    def loss_func(self):
        return nn.BCEWithLogitsLoss()
    
class LSTMRegression(nn.Module):
    def __init__(self, hps):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        out = self.sigmoid(out)
        return out
    
    def loss_func(self):
        return nn.MSELoss()
    
class LSTM_MSE(nn.Module):
    def __init__(self, hps):
        super(LSTM_MSE, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        out = self.sigmoid(out)
        return out
    
    def loss_func(self):
        return nn.MSELoss()