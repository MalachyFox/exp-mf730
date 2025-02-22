from dataloader import Manifest
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataclasses import dataclass
from pprint import pprint
import wandb
import os
from testing import do_test
from tools.analyse_results import analyse
from tqdm import tqdm

class CustomLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, hps):
        self.warmup_steps = hps.warmup_steps
        self.hold_steps = hps.hold_steps
        self.decay_steps = hps.num_epochs - hps.warmup_steps - hps.hold_steps
        self.total_steps = hps.num_epochs
        self.max_lr = hps.lr
        self.final_lr = hps.lr/100

        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # Linear warmup: increase from 0 to max_lr
            return step / self.warmup_steps
        elif step < self.warmup_steps + self.hold_steps:
            # Hold max_lr constant
            return 1.0
        else:
            # Exponential decay: decay to final_lr
            decay_progress = (step - self.warmup_steps - self.hold_steps) / self.decay_steps
            return (self.final_lr / self.max_lr) ** decay_progress  # Exponential decay formula


def do_train(model, train_dataloader, dev_dataloader, hps,testing=True):
    print('Training...')
    if hps.threads > 1 and hps.device == 'cpu':
        torch.set_num_threads(os.cpu_count()//hps.threads)
    wandb.init(
    project="speech-disorders",
    group = hps.group,
    name = hps.name,
    config = hps
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,weight_decay=hps.weight_decay)
    if hps.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=hps.scheduler_step_size,gamma=hps.scheduler_gamma)
    elif hps.scheduler == 'custom':
        scheduler = CustomLRScheduler(optimizer,hps)

    loss_function = model.loss_func()
    losses = []

    for epoch in range(hps.num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets, _ in tqdm(train_dataloader):
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

            wandb.log({"avg_loss": avg_loss,
                       "dev_loss": dev_loss,
                    "balanced_accuracy": balanced_accuracy,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    
                    })
        else:
            wandb.log({"avg_loss":avg_loss})
        losses.append(avg_loss)

        print(f"epoch {epoch+1:04}/{hps.num_epochs:04}, loss: {avg_loss:.4f}, dev_loss: {dev_loss:.4f}")
    wandb.finish()
    return losses

def save_checkpoint(model,hps,name='checkpoint'):
    checkpoint = {'model_state_dict': model.state_dict(), 'hps': hps}
    torch.save(checkpoint, f'./checkpoints/{hps.name}_{name}.pth')

