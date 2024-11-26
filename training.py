from dataloader.dataloader import Manifest
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint



def do_train(model, train_dataloader, hps,saving=False):

    optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr)

    loss_function = nn.BCEWithLogitsLoss()
    losses = []
    best_loss = 100
    for epoch in range(hps.num_epochs):
        model.train()
        total_loss = 0

        # preds = []
        # labels = []
        for inputs, targets in train_dataloader:

            inputs, targets = inputs.squeeze(), targets.squeeze()
            outputs = model(inputs).squeeze()
            # pred = torch.sigmoid(outputs)
            # preds.append(pred)
            # labels.append(targets)
            loss = loss_function(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        # # results = zip(labels, preds)
        # analyse_results(results)

        if saving:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model,hps,name='best')
            save_checkpoint(model,hps,name='epoch{epoch}')
        losses.append(avg_loss)
        print(f"epoch {epoch+1:04}/{hps.num_epochs:04}, loss: {avg_loss:.4f}")
        #plot_losses(losses,hps.name)
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
    return analysis


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

# if __name__ == "__main__":
#     name = 'test1'
#     resume = False

#     if resume == False:
#         ## initialization ##
#         hps = HyperParams(name)
#         pprint(hps)
#         model = LSTMClassifier(hps)
        
#         ## load data ##
#         train_dataloader, test_dataloader = load_data(hps)

#         ## train model ##
#         losses = do_train(model, train_dataloader, hps)

#         ## saving ##
#         plot_losses(losses, hps.name)
        
#         save_checkpoint(model,hps,'final')


        


