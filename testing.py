
import torch
import models
from hyperparams import HyperParams
import matplotlib.pyplot as plt
from pprint import pprint

def do_test(model,dataloader,hps):
    with torch.no_grad():
        results = []
        loss_func = model.loss_func()
        total_loss = 0
        for data, label, id in dataloader:
            data = data.to(hps.device)
            label = label.to(hps.device)
            data, label = data.squeeze(), label.squeeze()
            pred = model(data).squeeze().to(hps.device)

            total_loss += loss_func(pred,label)

            results.append((id[0],label.item(),pred.item()))

        avg_loss = total_loss/len(dataloader)
        return avg_loss, results