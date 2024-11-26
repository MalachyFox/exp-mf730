
from dataloader.dataloader import Manifest
from model import LSTMClassifier, HyperParams
from pprint import pprint
from training import do_train
from testing import do_test
import json

if __name__ == "__main__":
    hps = HyperParams.load_json('./hps_test2.json')
    pprint(hps)
    manifest = Manifest('./manifest.json')
    
    k = hps.k_fold
    results = []
    for i in range(k):
        hps.name = f'test2_{i}i_{k}k'
        train, test = manifest.get_k(i,k)
        model = LSTMClassifier(hps)
        losses = do_train(model,train,hps)
        result = do_test(model,test,hps)
        results.append({'fold':i,
                        'losses':losses,
                        'result': result})

    pprint(results)
    with open('./results_test2.json','w') as f:
        json.dump(results,f,indent=4)
        
