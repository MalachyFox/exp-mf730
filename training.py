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
import os
from testing import do_test
from tools.analyse_results import analyse

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def do_train(model, train_dataloader, dev_dataloader, hps,testing=True):
    print('Training...')
    if hps.threads > 1 and hps.device == 'cpu':
        torch.set_num_threads(os.cpu_count()//hps.threads)
    wandb.init(
    project="speech-disorders",
    group= f'{hps.group}_{timestamp()}',
    name = hps.name,
    config = hps
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,weight_decay=hps.weight_decay)
    if hps.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=hps.scheduler_step_size,gamma=hps.scheduler_gamma)
    loss_function = model.loss_func()
    losses = []

    for epoch in range(hps.num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets, _ in train_dataloader:
            inputs = inputs.to(hps.device)
            targets = targets.to(hps.device)

            outputs = model(inputs)

            outputs, targets = outputs.squeeze(), targets.squeeze()
            loss = loss_function(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        
        if hps.scheduler:
            scheduler.step()
        
        
        if testing:
            dev_loss, results = do_test(model,dev_dataloader,hps)
            analysis = analyse(results)
            balanced_accuracy = analysis['balanced_accuracy']
            sensitivity = analysis['sensitivity']
            specificity = analysis['specificity']

        else:
            dev_loss = None

        if testing:
            wandb.log({"avg_loss": avg_loss,
                       "dev_loss": dev_loss,
                    "balanced_accuracy": balanced_accuracy,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                    })
        else:
            wandb.log({"avg_loss":avg_loss})
        losses.append(avg_loss)

        print(f"epoch {epoch+1:04}/{hps.num_epochs:04}, loss: {avg_loss:.4f}, dev_loss: {dev_loss:.4f}")
        
    return losses

def save_checkpoint(model,hps,name='checkpoint'):
    checkpoint = {'model_state_dict': model.state_dict(), 'hps': hps}
    torch.save(checkpoint, f'./checkpoints/{hps.name}_{name}.pth')
