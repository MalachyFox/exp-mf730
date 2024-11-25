
from dataloader.dataloader import Manifest
from model import LSTMClassifier, HyperParameters
from pprint import pprint

if __name__ == "__main__":
    hps = HyperParameters.from_json('./hps.json')
    pprint(hps)

    k = 5
    for i in range(k):
        train, test = Manifest.get_k(i,k)
        model = LSTMClassifier(hps)