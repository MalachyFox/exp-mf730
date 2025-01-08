import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run analysis on results JSON file.")
    parser.add_argument("--test_name", required=True, help="Path to the results JSON file.")
    args = parser.parse_args()
    test_name = args.test_name # not including results_

    output_folder = f'./results/{test_name}/'
    os.makedirs(output_folder, exist_ok=True)
    with open(f'{output_folder}/results.json','r') as f:
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
        ids = fold['result']['ids']
        losses = fold['losses']
        for label, pred in zip(labels,preds):
            if label == 0:
                c00.append(pred)
            if label == 0.5:
                c05.append(pred)
            if label == 1.0:
                c10.append(pred)
        print(f'\nFold: {fold["fold"] + 1}')
        for i in range(len(labels)):
            print(ids[i][0],preds[i],labels[i])
        #print(labels,preds)
        plt.plot(losses)
        
        table = generate_table(labels, preds)
        analysis = analyse_table(table)
        balanced_accuracies.append(analysis['balanced_accuracy'])
        sensitivities.append(analysis['sensitivity'])
        specificities.append(analysis['specificity'])

        for key, value in analysis.items():
            print(f"{key}: {value}")
    
    combined_analysis = {   'balanced accuracy': np.mean(balanced_accuracies),
                            'sensitivity': np.mean(sensitivities),
                            'specificity': np.mean(specificities)}
    print()
    for key, value in combined_analysis.items():
        print(f"{key}: {value}")

    with open(f'{output_folder}/analysis.txt', "w") as f:
        json.dump(combined_analysis, f, indent=4)
        
    plt.savefig(f'{output_folder}/losses.png')
    plt.close()
    lens = np.array([len(c00),len(c05),len(c10)])
    plt.violinplot([c00,c05,c10],[0,0.5,1],widths=0.5* lens/np.max(lens))
    plt.xticks([0,0.5,1])
    plt.ylim(0,1)
    plt.xlabel('Actual Diagnosis')
    plt.ylabel('Prediced Diagnosis')
    plt.savefig(f'{output_folder}/violin.png')
