import json
import torch
import torch.nn as nn
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
        self.input_dropout = hps.input_dropout

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
        if self.input_dropout:
            x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x)

        if cell.dim() == 2:
            cell = cell.unsqueeze(1)

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
            return nn.MSELoss()


class AttentionBlockModel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.device = hps.device
        self.loss_function = hps.loss_function
        self.block_size = hps.block_size
        self.num_heads = hps.num_heads

        self.attention = nn.MultiheadAttention(hps.input_size, hps.num_heads, batch_first=True)
        self.final_attention = nn.MultiheadAttention(hps.input_size,hps.num_heads, batch_first=True)
        self.final_output = nn.Linear(hps.input_size, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)  # Move input to the same device as model
        B, T, D = x.shape  # Batch, Sequence Length, Feature Dim
        
        # # Pad sequence length to be a multiple of block_size
        # pad_len = (self.block_size - (T % self.block_size)) % self.block_size
        # if pad_len > 0:
        #     pad_tensor = torch.zeros((B, pad_len, D), device=x.device)
        #     x = torch.cat([x, pad_tensor], dim=1)
        #        x = x[:,:-1,:,:]
        
        # Reshape into blocks
        num_blocks = x.shape[1] // self.block_size
        x = x[:,:num_blocks*self.block_size,:]
        x = x.reshape(B, num_blocks, self.block_size, -1)
        
        x = x.reshape(num_blocks,self.block_size,-1)
        out, _ = self.attention(x, x, x)
        out = torch.mean(out,axis=1)
        print(out.shape)

        # Apply attention to each block
        # block_outputs = []
        # for i in range(num_blocks - 1):
        #     block = x[:, i, :, :]
        #     out, _ = self.attention(block, block, block) # B, L, E - B, L, L
        #     out = torch.mean(out,dim=1).reshape(B,1,-1)
        #     # weights = torch.mean(weights,dim=1) # B, L
        #     # out = torch.matmul(weights,out) # computes batched weighted sum
        #     block_outputs.append(out)
        
        # # Concatenate all block scalars
        # out = torch.cat(block_outputs,axis=1)  # Shape: (B, num_blocks_2, mid_size)

        # Apply second stage attention
        out, _ = self.final_attention(out, out, out) # B, L', E
        # weights = torch.mean(weights,dim=1) # B, L
        # out = torch.matmul(weights,out).reshape(B,-1) # computes batched weighted sum
        out = torch.mean(out,dim=1)
        out = self.final_output(out)  # Final scalar output shape (B,1)
        out = self.softmax(out)
        out = out[:,0]
        return out  # Shape: (B,)
    
    def loss_func(self):
        if self.loss_function == 'bce':
            return nn.BCELoss()
        if self.loss_function == 'mse':
            return nn.MSELoss()