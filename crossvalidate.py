
from dataloader.dataloader import Manifest
from model import LSTMClassifier, HyperParams
from pprint import pprint
from training import do_train
from testing import do_test
import json

if __name__ == "__main__":
    hps = HyperParams.load_json('./hps_test2.json')
    pprint(hps)

    manifest = Manifest(hps.manifest_path)
    k = hps.k_fold
    name = hps.name

    results = []
    for i in range(k):
        hps.name = f'{name}_{i}i_{k}k' # change this probably
        train, test = manifest.get_k(i,k)
        model = LSTMClassifier(hps)
        losses = do_train(model,train,hps)
        result = do_test(model,test,hps)
        results.append({'fold':i,
                        'losses':losses,
                        'result': result})

    pprint(results)
    with open(f'./results_{name}.json','w') as f:
        json.dump(results,f,indent=4)
        
