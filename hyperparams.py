from dataclasses import dataclass, asdict
import json
import torch
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

@dataclass
class HyperParams():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.group = self.name + '_' + timestamp()
        self.ensemble_id = 0
    @classmethod
    def load_json(self,path):
        with open(path) as f:
            data = json.load(f)
            return self(**data)

    def save_json(self,path):
        with open(path, "w") as json_file:
            json.dump(asdict(self), json_file, indent=4)
    
    def __repr__(self):
        out = ""
        out += "HyperParams:\n"
        for key, value in self.__dict__.items():
            out+= f"  {key}: {value}\n"
        return out