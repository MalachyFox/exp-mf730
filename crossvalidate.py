
from dataloader import Manifest
import models
from hyperparams import HyperParams
from pprint import pprint
from training import do_train
from testing import do_test
import json
import argparse
import os
from multiprocessing import Pool

def load_model(hps):
    model_name = hps.model
    model_class = getattr(models,model_name)
    model = model_class(hps)
    return model

def crossvalidation_task(args):
    i,k,manifest,hps,name = args
    model = load_model(hps).to(hps.device)
    
    hps.name = f'{name}_{i+1}i_{k}k' # change this probably
    
    # Folded dataloader
    train, test = manifest.get_k(i,hps)
    
    # Train
    losses = do_train(model,train, test, hps)

    test_loss, results = do_test(model,test,hps)
    # Test 
    return results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run crossvalidated training and testing pipeline.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load hyperparameters from JSON
    hps = HyperParams.load_json(args.config)
    manifest = Manifest(hps.manifest_path)
    k = hps.k_fold
    name = hps.name
    
    pprint(hps)

    results_list = []
    args_list = [(i, k, manifest, hps, name) for i in range(k)]

    if hps.threads < 2:
        for arg in args_list:
            results_list.append(crossvalidation_task(arg))

    else:
        # Run crossvalidation
        with Pool(processes=hps.threads) as pool:
            results_list = pool.map(crossvalidation_task, args_list)
        
    results = []
    [results.extend(r) for r in results_list]

    # Save results
    results_folder = f'./results/{name}'
    results_path = f'{results_folder}/results.json'
    os.makedirs(results_folder, exist_ok=True)
    print(f'Saving results to {results_path}')
    with open(results_path,'w') as f:
        json.dump(results,f,indent=4)
        
