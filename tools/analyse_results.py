import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

def generate_table(results):
    labels = [r[1] for r in results]
    preds = [r[3] for r in results]
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

def add_clamped(results):
    out = []
    for r in results:
        id, label, pred = r
        clamped = pred >= 0.5
        out.append((id,label,pred,clamped))
    return out

def analyse(results):

    results = add_clamped(results)
    table = generate_table(results)
    analysis = analyse_table(table)
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


    analysis = analyse(results)


    with open(f'{output_folder}/statistics.txt', "w") as f:
        json.dump(analysis, f, indent=4)

    results = np.array(results)
    diffs = np.array([abs(float(a[2]) - float(a[1])) for a in results])
    sorted_indices = diffs.argsort()
    sorted_predictions = results[sorted_indices]
    
    np.savetxt(f'{output_folder}/predictions.txt', sorted_predictions, fmt='%s')

    c00 = [float(r[2]) for r in results if r[1] == '0.0']
    c05 = [float(r[2]) for r in results if r[1] == '0.5']
    c10 = [float(r[2]) for r in results if r[1] == '1.0']

    lens = np.array([len(c00),len(c05),len(c10)])
    plt.violinplot([c00,c05,c10],[0,0.5,1],widths=0.5* lens/np.max(lens))
    plt.xticks([0,0.5,1])
    plt.ylim(0,1)
    plt.xlabel('Actual Diagnosis')
    plt.ylabel('Prediced Diagnosis')
    plt.savefig(f'{output_folder}/violin.png')
