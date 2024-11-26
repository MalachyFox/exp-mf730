
import torch
from model import LSTMClassifier, HyperParams
import matplotlib.pyplot as plt
from pprint import pprint

def get_results(model,dataloader):
    with torch.no_grad():
        labels = []
        preds = []
        for data, label in dataloader:
            data, label = data.squeeze(), label.squeeze().item()
            output = model(data)
            pred = torch.sigmoid(output).squeeze().item()
            preds.append(pred)
            labels.append(label)
        plt.scatter(labels,preds)
        plt.savefig('./plots/testing_recent.png')
        return {'labels': labels,
                'preds': preds}

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
    results = get_results(model,test_dataloader)
    # table = generate_table(results)
    # analysis = analyse_table(results)
    return results

if __name__ == "__main__":
    pass
    # name = 'test1'
    # ## load checkpoint ##
    # checkpoint_path = f'./checkpoints/{name}_best.pth'
    # print(f'Loading checkpoint {checkpoint_path}...')
    # checkpoint = torch.load(checkpoint_path)

    # ## load hyperparameters ##
    # print('Hyperparameters:')
    # hps = checkpoint['hps']
    # pprint(hps)

    # ## initialize model ##
    # model = LSTMClassifier(hps)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # ## load data ##
    # train_dataloader, test_dataloader = load_data(hps)

    # ## run test ##
    # get_results(model,test_dataloader)