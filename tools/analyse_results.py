import json
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    with open('./results/results_test2.json','r') as f:
        results = json.load(f)

    balanced_accuracies = []
    sensitivities = []
    specificities = []
    c10 = []
    c05 = []
    c00 = []
    for fold in results:
        labels = fold['result']['labels']
        preds = fold['result']['preds']
        losses = fold['losses']
        for label, pred in zip(labels,preds):
            if label == 0:
                c00.append(pred)
            if label == 0.5:
                c05.append(pred)
            if label == 1.0:
                c10.append(pred)
        print(labels,preds)
        plt.plot(losses)
        print(fold["fold"])
        table = generate_table(labels, preds)
        analysis = analyse_table(table)
        balanced_accuracies.append(analysis['balanced_accuracy'])
        sensitivities.append(analysis['sensitivity'])
        specificities.append(analysis['specificity'])
        print(analysis)
    print(np.mean(balanced_accuracies))
    print(np.mean(sensitivities))
    print(np.mean(specificities))
    plt.savefig('./plots/losses.png')
    plt.close()
    lens = np.array([len(c00),len(c05),len(c10)])
    plt.violinplot([c00,c05,c10],[0,0.5,1],widths=0.5* lens/np.max(lens))
    plt.xticks([0,0.5,1])
    plt.xlabel('Actual Diagnosis')
    plt.ylabel('Prediced Diagnosis')
    plt.savefig('./plots/violin.png')
