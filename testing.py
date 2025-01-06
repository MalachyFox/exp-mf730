
import torch
import models
from hyperparams import HyperParams
import matplotlib.pyplot as plt
from pprint import pprint

def get_results(model,dataloader):
    with torch.no_grad():
        labels = []
        preds = []
        ids = []
        for data, label, id in dataloader:
            data, label = data.squeeze(), label.squeeze().item()
            output = model(data)
            pred = torch.sigmoid(output).squeeze().item()
            preds.append(pred)
            labels.append(label)
            ids.append(id)
        plt.scatter(labels,preds)
        plt.savefig('./plots/testing_recent.png')
        return {'labels': labels,
                'preds': preds,
                'ids':ids}

def do_test(model,test_dataloader,hps):
    print('Testing...')
    results = get_results(model,test_dataloader)
    return results
