# average length
# proportion adult speaking
# proportion of diagnosis
# proportion of errors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import os
import numpy as np
from matplotlib import pyplot as plt
from dataloader.dataloader import ManifestEntry

from pprint import pprint


def load_manifest(path="./manifest.json"):
    manifest_path = './manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    for i, m in enumerate(manifest):
        m = ManifestEntry.from_json(m)
        manifest[i] = m
    return manifest


if __name__ == "__main__":
    manifest_path = './manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    num_total = len(manifest)
    errors = 0
    new_manifest = []
    for i,m in enumerate(manifest):
        if m['diagnosis'] == None:
            errors += 1
        else:
            new_manifest.append(m)
    manifest = new_manifest # removes nulls
    num = num_total - errors
    print(f'\nnumber of recordings: {num_total}')
    print(f'recordings with valid diagnosis data: {num}')

    ##

    durations = [m['duration'] for m in manifest]
    average_length = np.mean(durations)
    plt.hist(durations,bins = 20)
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.title('Histogram of durations of recordings')
    plt.savefig('./plots/recording_duration_histogram.png')
    print('\nof complete recordings:')
    print(f'    average duration: {np.mean(durations):.1f}s')
    print(f'    combined duration: {np.sum(durations)/(60*60):.1f}h')

    ##

    scores = [m['diagnosis'] for m in manifest]
    score, count = np.unique(scores, return_counts=True)
    print()
    for s, c in zip(score,count):
        print(f'{c}/{num} have diagnosis {s}')

    ## 

    durs = []
    for i, m in enumerate(manifest):
        m = ManifestEntry.from_json(m)
        manifest[i] = m
        child_ortho_data = m.child_ortho_data
        durs.extend([parts[1] for parts in child_ortho_data]) # takes duration info from data
    mean_dur = np.mean(durs)
    durs.sort()
    #print(f'\nlongest continuous child section durations: {durs[-10:]}')
    utterance_upper_bound = 5 # number of seconds of utterance
    count = sum([1 for d in durs if d > utterance_upper_bound])
    count_100 = sum([1 for d in durs if d > 100])
    print(f'\n{count} utterances longer than {utterance_upper_bound}s')
    print(f'{count_100} utterances longer than 100s')

    print(f'\nnumber of utterances: {len(durs)}')
    print(f'excluded {count}/{len(durs)} utterances longer than {utterance_upper_bound}s')
    durs = [d for d in durs if d <= utterance_upper_bound]
    
    print('\nof utterances:')
    print(f'    average duration: {np.mean(durs):.1f}s')
    print(f'    combined duration: {np.sum(durs)/(60*60):.1f}h')

    
    plt.figure()
    plt.hist(durs,bins=50)  
    plt.title('Histogram of durations of utterances of continuous child speech')
    plt.xlabel('Duration of section (s)')
    plt.ylabel('Count')
    plt.savefig('./plots/child_seperated_utterance_durations.png')


    