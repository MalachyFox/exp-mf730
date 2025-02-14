import json
import torch
import torch.nn as nn
import torch.nn.functional
from hyperparams import HyperParams
    
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.bidirectional = hps.bidirectional
        self.hidden_size = hps.hidden_size
        self.batch_size = hps.batch_size
        num_directions = 2 if self.bidirectional else 1
        self.loss_function = hps.loss_function

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.dropout(x)
        
        output, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            out = torch.cat((cell[-2], cell[-1]), dim=1).reshape(self.batch_size,2 * self.hidden_size)
        else:
            out = cell[-1].reshape(self.batch_size,self.hidden_size)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

    def loss_func(self):
        if self.loss_function == 'bce':
            return nn.BCELoss()
        if self.loss_function == 'mse':
            return torch.nn.functional.mse_loss


class AttentionClassTokenModel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.device = hps.device
        self.loss_function = hps.loss_function
        self.block_size = hps.block_size
        self.input_size = hps.input_size
        self.num_heads = hps.num_heads

        self.dropout = nn.Dropout(hps.dropout)
        self.attention = nn.MultiheadAttention(hps.input_size, hps.num_heads, batch_first=True,dropout=hps.dropout)
        self.class_token = nn.Parameter(torch.randn(1,1,hps.input_size))
        self.class_token2 = nn.Parameter(torch.randn(1,1,hps.input_size))
        self.final_attention = nn.MultiheadAttention(hps.input_size,hps.num_heads, batch_first=True,dropout=hps.dropout)
        self.linear = nn.Linear(hps.input_size, hps.linear_size)
        self.nonlinear = nn.ReLU()
        self.final_linear = nn.Linear(hps.linear_size, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        B, T, D = x.shape  # Batch, Sequence Length, Feature Dim
        
        # Reshape into blocks
        num_blocks = x.shape[1] // self.block_size
        x = x[:,:num_blocks*self.block_size,:] # trim to whole number of blocks
        x = x.reshape(num_blocks,self.block_size,self.input_size) # num_blocks, block_size, input size

        x = self.dropout(x)

        # add class tokens
        class_tokens = self.class_token.expand((num_blocks,1,self.input_size))
        x = torch.cat((class_tokens,x),dim=1)

        out, _ = self.attention(x, x, x)

        out = out[:,0,:] # num_blocks, input_size
        out = out.reshape((1,num_blocks,self.input_size))# 1, num_blocks, input_size
        
        out = torch.cat((self.class_token2,out),dim=1)
        
        out, _ = self.final_attention(out, out, out)
        out = out[:,0,:] # 1, input_size

        out = self.linear(out)
        out = self.nonlinear(out)
        out = self.dropout(out)

        out = self.final_linear(out)
        out = self.softmax(out)
        out = out[:,0]
        return out  # Shape: (B,)
    
    def loss_func(self):
        if self.loss_function == 'bce':
            return nn.BCELoss()
        if self.loss_function == 'mse':
            return torch.nn.functional.mse_loss


class AttentionBlockModel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.device = hps.device
        self.loss_function = hps.loss_function
        self.block_size = hps.block_size
        self.num_heads = hps.num_heads

        self.dropout = nn.Dropout(hps.dropout)
        self.attention = nn.MultiheadAttention(hps.input_size, hps.num_heads, batch_first=True,dropout=hps.dropout)
        self.linearmid = nn.Linear(hps.input_size,hps.input_size)
        self.final_attention = nn.MultiheadAttention(hps.input_size,hps.num_heads, batch_first=True,dropout=hps.dropout)
        self.linear = nn.Linear(hps.input_size, hps.linear_size)
        self.relu = nn.ReLU()
        self.final_output = nn.Linear(hps.linear_size, 2)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        B, T, D = x.shape  # Batch, Sequence Length, Feature Dim
        
        # Reshape into blocks
        num_blocks = x.shape[1] // self.block_size
        x = x[:,:num_blocks*self.block_size,:]

        #remove batch
        x = x.reshape(num_blocks,self.block_size,-1)

        x = self.dropout(x)
        out, _ = self.attention(x, x, x)
        out = torch.mean(out,axis=1)
        out, _ = self.final_attention(out, out, out) # B, L', E
        out = torch.mean(out,axis=0)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.final_output(out)  # Final scalar output shape (B,1)
        out = self.softmax(out)
        out = out[0]
        return out  # Shape: (B,)
    
    def loss_func(self):
        if self.loss_function == 'bce':
            return nn.BCELoss()
        if self.loss_function == 'mse':
            return torch.nn.functional.mse_loss