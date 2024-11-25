from dataloader.dataloader import Manifest
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint


class LSTMClassifier(nn.Module):
    def __init__(self, hps):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(hps.input_size, hps.hidden_size, hps.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hps.hidden_size,out_features=1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.linear(cell)
        return out

def do_train(model, train_dataloader, optimizer, scheduler, device, hps):
    loss_function = nn.BCEWithLogitsLoss()
    losses = []
    best_loss = 100
    for epoch in range(hps.num_epochs):
        model.train()
        total_loss = 0

        preds = []
        labels = []
        for inputs, targets in train_dataloader:

            inputs, targets = inputs.to(device).squeeze(), targets.to(device).squeeze()
            outputs = model(inputs).squeeze()
            pred = torch.sigmoid(outputs)
            preds.append(pred)
            labels.append(targets)
            loss = loss_function(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        results = zip(labels, preds)
        analyse_results(results)


        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model,hps,name='best')
        losses.append(avg_loss)
        print(f"epoch {epoch+1:04}/{hps.num_epochs:04}, loss: {avg_loss:.4f}")
        scheduler.step()
        
    return losses


def analyse_results(results):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for label, pred in results:
        if label == 0.5:
            continue
        else:
            if label == 1:
                if pred > 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred > 0.5:
                    FP += 1
                else:
                    TN += 1
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    analysis = {'sensitivity': sensitivity,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy}
    pprint(analysis)
    return analysis

@dataclass
class HyperParams():
    name: str
    batch_size : int = 1
    lr: float = 0.001
    num_epochs: int = 20
    input_size: int = 768
    hidden_size: int = 256
    num_layers: int = 1
    num_testing: int = 20
    lrs_step_size: int = 5
    lrs_gamma: float = 0.5


class ListDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels_list[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def load_data(hps):
    data = Manifest()

    embeddings = data.get_wav2vec2_embeddings()
    labels = data.labels

    train_embs = embeddings[:-hps.num_testing]
    train_labels = labels[:-hps.num_testing]
    train_dataset = ListDataset(train_embs,train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size = hps.batch_size, shuffle = True)

    test_embs = embeddings[-hps.num_testing:]
    test_labels = labels[-hps.num_testing:]
    test_dataset = ListDataset(test_embs,test_labels)
    test_dataloader = DataLoader(test_dataset,batch_size = hps.batch_size, shuffle = True)

    return train_dataloader, test_dataloader


def plot_losses(losses,name):
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(f'./plots/train_losses_{name}.png')

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def save_checkpoint(model,hps,name='checkpoint'):
    checkpoint = {'model_state_dict': model.state_dict(), 'hps': hps}
    torch.save(checkpoint, f'./checkpoints/{hps.name}_{name}.pth')

if __name__ == "__main__":
    name = 'test1'
    resume = False

    if resume == False:
        ## initialization ##
        hps = HyperParams(name)
        pprint(hps)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMClassifier(hps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=hps.lrs_step_size,gamma=hps.lrs_gamma)

        ## load data ##
        train_dataloader, test_dataloader = load_data(hps)

        ## train model ##
        losses = do_train(model, train_dataloader, optimizer, scheduler, device, hps)

        ## saving ##
        plot_losses(losses, hps.name)
        
        save_checkpoint(model,hps,'final')


        


