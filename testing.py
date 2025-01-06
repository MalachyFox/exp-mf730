
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

def generate_table(labels, preds):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for label, pred in zip(labels,preds):
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
    return (TP, FP, TN, FN)

def analyse_table(table):
    TP, FP, TN, FN = table
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    analysis = {'sensitivity': sensitivity,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy}
    return analysis

def do_test(model,test_dataloader,hps):
    print('Testing...')
    results = get_results(model,test_dataloader)
    # table = generate_table(results)
    # analysis = analyse_table(results)
    return results
