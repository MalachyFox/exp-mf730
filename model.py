import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict

@dataclass
class HyperParams():
    name: str
    batch_size : int = 1
    lr: float = 0.001
    num_epochs: int = 20
    input_size: int = 768
    hidden_size: int = 256
    num_layers: int = 1
    lrs_step_size: int = 5
    lrs_gamma: float = 0.5
    k_fold: int = 5

    @classmethod
    def load_json(self,path):
        with open(path) as f:
            data = json.load(f)
            return self(**data)

    def save_json(self,path):
        with open(path, "w") as json_file:
            json.dump(asdict(self), json_file, indent=4)


class LSTMClassifier(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        return out
    
if __name__ == '__main__':
    pass
    # hps = HyperParams('test2')
    # hps.save_json('./hps.json')