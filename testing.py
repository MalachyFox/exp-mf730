
import torch
from training import LSTMClassifier, load_data, HyperParams, timestamp
import matplotlib.pyplot as plt
from pprint import pprint

def do_test(model,dataloader):
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
        plt.savefig('./plots/testing.png')
        return zip(labels,preds)

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

def get_results(model,test_dataloader):
    results = do_test(model,test_dataloader)
    analysis = analyse_results(results)
    return analysis

if __name__ == "__main__":
    name = 'test1'
    ## load checkpoint ##
    checkpoint_path = f'./checkpoints/{name}_best.pth'
    print(f'Loading checkpoint {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path)

    ## load hyperparameters ##
    print('Hyperparameters:')
    hps = checkpoint['hps']
    pprint(hps)

    ## initialize model ##
    model = LSTMClassifier(hps)
    model.load_state_dict(checkpoint['model_state_dict'])

    ## load data ##
    train_dataloader, test_dataloader = load_data(hps)

    ## run test ##
    get_results(model,test_dataloader)
