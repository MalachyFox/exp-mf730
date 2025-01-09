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
        cell_top = cell[-1]
        out = self.linear(cell_top)
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
        return out
    
    def loss_func(self):
        return nn.MSELoss()
    
class LSTM_MSE(nn.Module):

    def __init__(self, hps):
        super(LSTM_MSE, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        out = self.sigmoid(out)
        return out
    
    def loss_func(self):
        return nn.MSELoss()
    
class LSTMClassifier2(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifier2, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear1 = nn.Linear(in_features=hps.hidden_size,out_features=hps.linear_size)
        self.linear2 = nn.Linear(in_features=hps.linear_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = cell[-1]
        out = self.linear1(out)
        out = self.linear2(out)
        return out
    
    def loss_func(self):
        return nn.BCEWithLogitsLoss()
    
class LSTMClassifierBi(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifierBi, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True,bidirectional=True)
        self.linear1 = nn.Linear(in_features=hps.hidden_size*2,out_features=hps.linear_size)
        self.linear2 = nn.Linear(in_features=hps.linear_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = cell.flatten()
        out = self.linear1(out)
        out = self.linear2(out)
        return out
    
    def loss_func(self):
        return nn.BCEWithLogitsLoss()

class LSTMClassifierBiLarge(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifierBiLarge, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True,bidirectional=True)
        self.linear1 = nn.Linear(in_features=hps.hidden_size*2,out_features=hps.linear_size)
        self.linear2 = nn.Linear(in_features=hps.linear_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = cell[-2:].flatten()
        out = self.linear1(out)
        out = self.linear2(out)
        return out
    
    def loss_func(self):
        return nn.BCEWithLogitsLoss()