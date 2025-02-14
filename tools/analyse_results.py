import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc

def generate_table(results):
    labels = [r[1] for r in results]
    clamped_preds = [r[3] for r in results]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for label, pred in zip(labels,clamped_preds):
        if label == 0.5:
            continue
        else:
            if label == 1 and pred == 1:
                    TP += 1
            elif label == 1 and pred == 0:
                    FN += 1
            elif label == 0 and pred == 1:
                    FP += 1
            elif label == 0 and pred == 0:
                    TN += 1
            else:
                print(label, pred)
                raise ValueError
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



def clamp(results,threshold):
    output = []
    for r in results:
        id, label, pred = r
        if pred >= threshold:
            clamped = 1.0
        else:
            clamped = 0.0
        output.append([id, label, pred, clamped])
    return output

def analyse(results,threshold=0.5):
    clamped_results = clamp(results,threshold)
    table = generate_table(clamped_results)
    analysis = analyse_table(table)
    return analysis

def analyse_ensemble(results,threshold=0.5):
    clamped_results = clamp(results,threshold)
    clamped_results = ensemble_combine(clamped_results)
    table = generate_table(clamped_results)
    analysis = analyse_table(table)
    return analysis

def seperate_results(results,ensemble_num = 5):
    if len(results) % ensemble_num != 0:
        raise ValueError
    size = len(results) // ensemble_num
    results_list = []
    for i in range(ensemble_num):
         results_list.append(results[i*size:(i+1)*size])
    return results_list

def analyse_seperated(results,threshold=0.5):
    results_list = seperate_results(results)
    analysis_list = []
    for results in results_list:
         analysis_list.append(analyse(results,threshold))
    
    sens = [a['sensitivity'] for a in analysis_list]
    spec = [a['specificity'] for a in analysis_list]
    ba = [a['balanced_accuracy'] for a in analysis_list]
    output = []
    for item in [sens,spec,ba]:
        mean = np.mean(item)
        output.append((mean,np.min(item)-mean,np.max(item)-mean))
    return {
         'sensitivity': output[0],
         'specificity': output[1],
         'balanced_accuracy': output[2]
    }

def ensemble_combine(results):
    output = []
    ids = []
    for r in results:
        id = r[0]
        if id not in ids:
            ids.append(id)

    for id in ids:
        label = 0
        preds = []
        clamped_preds = []
        for r in results:
            if r[0] == id:
                label = r[1]
                preds.append(r[2])
                clamped_preds.append(r[3])
        
        pred = np.mean(preds)
        
        count = 0
        for p in clamped_preds:
             if p == 1.0:
                  count += 1

        if count >= len(clamped_preds)/2:
             clamped_pred = 1.0
        else:
             clamped_pred = 0.0

        output.append([id,label,pred,clamped_pred])
    return output
    
def full_analysis(results):
    analysis_sep = analyse_seperated(results,threshold=0.5)
    analysis_ensemble = analyse_ensemble(results,threshold=0.5)
    analysis = analyse(results)
    final_analysis = {'seperated': analysis_sep,'ensemble':analysis_ensemble}
    return final_analysis

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

    final_analysis = full_analysis(results)

    with open(f'{output_folder}/statistics.txt', "w") as f:
        json.dump(final_analysis, f, indent=4)


    ## GENERATE SORTED PREDICTIONS FILE ##
    
    results_array = np.array(results)
    diffs = np.array([abs(float(a[2]) - float(a[1])) for a in results_array])
    sorted_indices = diffs.argsort()
    sorted_predictions = results_array[sorted_indices]
    
    np.savetxt(f'{output_folder}/predictions.txt', sorted_predictions, fmt='%s')

    ## GENERATE VIOLIN ##

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
