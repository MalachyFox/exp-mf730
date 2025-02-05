import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc

def generate_table(results,threshold=0.5):
    labels = [r[1] for r in results]
    preds = [r[2] for r in results]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for label, pred in zip(labels,preds):
        if label == 0.5:
            continue
        else:
            if label == 1:
                if pred >= threshold:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred >= threshold:
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

def analyse(results,threshold=0.5):
    table = generate_table(results,threshold=threshold)
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

    results_array = np.array(results)
    diffs = np.array([abs(float(a[2]) - float(a[1])) for a in results_array])
    sorted_indices = diffs.argsort()
    sorted_predictions = results_array[sorted_indices]
    
    np.savetxt(f'{output_folder}/predictions.txt', sorted_predictions, fmt='%s')

    c00 = [float(r[2]) for r in results_array if r[1] == '0.0']
    c05 = [float(r[2]) for r in results_array if r[1] == '0.5']
    c10 = [float(r[2]) for r in results_array if r[1] == '1.0']

    lens = np.array([len(c00),len(c05),len(c10)])
    plt.violinplot([c00,c05,c10],[0,0.5,1],widths=0.5* lens/np.max(lens))
    plt.xticks([0,0.5,1])
    plt.ylim(0,1)
    plt.xlabel('Actual Diagnosis')
    plt.ylabel('Prediced Diagnosis')
    plt.savefig(f'{output_folder}/violin.png')
    plt.close()




    ### generate ROC ##

    n = 1000
    thresholds = np.linspace(0,1,n)
    tps = []
    fns = []
    for t in thresholds:
        analysis = analyse(results,threshold=t)
        true_positive_rate = analysis['sensitivity']
        false_negative_rate = 1 - analysis['specificity']
        tps.append(true_positive_rate)
        fns.append(false_negative_rate)
    roc_auc = auc(fns, tps)

    # Find the optimal threshold (closest to (0,1))
    optimal_idx = np.argmin(np.sqrt(np.array(fns) ** 2 + (1 - np.array(tps)) ** 2))
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(8, 8))

    # Highlight optimal threshold point
    plt.scatter(fns[optimal_idx], tps[optimal_idx], color='green', s=100, label=f'Optimal threshold {optimal_threshold:.2f} (TP: {tps[optimal_idx]:.2f}, FN: {fns[optimal_idx]:.2f})')
    plt.scatter(fns[n//2], tps[n//2], color='black', s=50,label=f'Threshold {0.5:.2f} (TP: {tps[n//2]:.2f}, FN: {fns[n//2]:.2f})')
    # Plot ROC Curve

    plt.step(fns, tps, color='blue', lw=2, label=f'ROC curve. Area = {roc_auc:.2f}')

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random classifier')

    # Labels, legend, and limits
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("False Negative Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{output_folder}/roc.png", dpi=300)
