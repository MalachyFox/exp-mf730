from dataloader import Manifest
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
import wandb


def do_train(model, train_dataloader, hps,saving=False):
    print('Training...')

    wandb.init(
    project="speech-disorders",
    config = hps
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr)

    loss_function = model.loss_func()
    losses = []

    for epoch in range(hps.num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets, _ in train_dataloader:

            outputs = model(inputs)
            outputs, targets = outputs.squeeze(), targets.squeeze()
            loss = loss_function(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)

        losses.append(avg_loss)
        print(f"epoch {epoch+1:04}/{hps.num_epochs:04}, loss: {avg_loss:.4f}")
        wandb.log({"loss": loss},step=epoch)
    return losses


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def save_checkpoint(model,hps,name='checkpoint'):
    checkpoint = {'model_state_dict': model.state_dict(), 'hps': hps}
    torch.save(checkpoint, f'./checkpoints/{hps.name}_{name}.pth')
