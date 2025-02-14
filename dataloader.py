import json
import soundfile as sf
import numpy as np
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.create_manifest import get_orthographic_data
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoProcessor, WavLMModel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import opensmile
from multiprocessing import Pool
from tqdm import tqdm


def extract_egemaps_features(args):
    segment, smile = args
    features = smile.process_signal(segment,sampling_rate = 16000)
    features = torch.tensor(features.to_numpy(),dtype=torch.float32)
    features = features.unsqueeze(0)
    return features

class Manifest:
    def __init__(self,manifest_path='/research/milsrg1/sld/exp-mf730/manifest.json'):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        manifest  = [ManifestEntry.from_json(m) for m in manifest]
        self.entries  = [m for m in manifest if m is not None]
        self.labels = [m.diagnosis for m in self.entries]
        self.manifest_path = manifest_path
        self.ids = [m.name for m in self.entries]

        N = len(self.labels)
        self.indices = np.arange(N)
        np.random.seed(0)
        np.random.shuffle(self.indices)
        

    def get_wav2vec2_embeddings(self, model_name = "wav2vec2-base", embeddings_folder = '/research/milsrg1/sld/exp-mf730/embeddings'):
        embedding_file = f'{model_name}_embeddings.pt'
        path = f'{embeddings_folder}/{embedding_file}'
        if os.path.exists(path):
            print(f'\nLoading embeddings for {self.manifest_path} from {path}\n')
            embeddings = torch.load(path)
        else:
            print(f'\ngenerating embeddings for {self.manifest_path} using {model_name} and saving to {embeddings_folder}\n')
            
            if model_name == 'wav2vec2-base':
                model_id = f'facebook/{model_name}'
                processor = Wav2Vec2Processor.from_pretrained(model_id)
                model = Wav2Vec2Model.from_pretrained(model_id)
            elif model_name == 'wavlm-base':
                processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
                model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")


            embeddings = []
            for entry in self.entries:
                print(f'\nprocessing entry {entry.name}')
                embeddings_list = []
                i = 0
                for segment in entry.child_segments:
                    print(f'processing segment {i:04}/{len(entry.child_segments):04}...',end='\r')
                    i += 1
            
                    input = processor(segment, return_tensors="pt", sampling_rate=entry.sample_rate).input_values # (1, num_samples)
                    
                    samples = input.shape[1]
                    min_samples = 512 
                    if samples < min_samples:
                        padding = min_samples - samples
                        input = F.pad(input,(0,padding))

                    with torch.no_grad():
                        output = model(input)
                    output_embeddings = output.last_hidden_state
                    embeddings_list.append(output_embeddings)

                if len(embeddings_list) < 2:
                    print('ERROR: no embeddings')
                    raise ValueError
                
                concatenated_embeddings = torch.cat(embeddings_list,dim = 1)
                embeddings.append(concatenated_embeddings)
                
            torch.save(embeddings,path)

        return embeddings
    
    def get_egemaps_embeddings(self,model_name="egemaps",embeddings_folder = '/research/milsrg1/sld/exp-mf730/embeddings',parallel=True):
        embedding_file = f'{model_name}_embeddings.pt'
        path = f'{embeddings_folder}/{embedding_file}'
        if os.path.exists(path):
            print(f'\nLoading embeddings for {self.manifest_path} from {path}\n')
            embeddings = torch.load(path)
        
        else:
            smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02)
            embeddings = []
            for entry in tqdm(self.entries):
                print(f'\nprocessing entry {entry.name}')
                embeddings_list = []
                i = 0
                segments = entry.child_segments
                segments = [s for s in segments if len(s) >= 1024]

                if parallel:
                    args = [(s,smile) for s in segments]
                    with Pool(processes=None) as pool:
                        embeddings_list = pool.map(extract_egemaps_features, args)
                else:
                    for segment in entry.child_segments:
                        print(f'processing segment {i:04}/{len(entry.child_segments):04}...',end='\r')
                        i += 1
                        features = smile.process_signal(segment,sampling_rate = entry.sample_rate)
                        features = torch.tensor(features.to_numpy(),dtype=torch.float32)
                        features = features.unsqueeze(0)
                        embeddings_list.append(extract_egemaps_features(segment,smile))

                if len(embeddings_list) < 2:
                    print('ERROR: no embeddings')
                    raise ValueError
                
                concatenated_embeddings = torch.cat(embeddings_list,dim = 1)
                embeddings.append(concatenated_embeddings)
            torch.save(embeddings,path)

        return embeddings
    
    def get_embeddings(self,model_name,embeddings_folder = '/research/milsrg1/sld/exp-mf730/embeddings'):
        if model_name == 'wav2vec2-base':
            return self.get_wav2vec2_embeddings(model_name,embeddings_folder)
        elif model_name == 'wavlm-base':
            return self.get_wav2vec2_embeddings(model_name,embeddings_folder)
        elif model_name == 'egemaps':
            return self.get_egemaps_embeddings(model_name, embeddings_folder)
    
    def get_k(self,i,hps):
        k = hps.k_fold
        print(f'Running cross validation {i+1}/{k}')
        N = len(self.labels)
        indices = self.indices
        test_size = N//k
        extra = N % k
        start = i * test_size + min(i, extra)
        end = start + test_size + (1 if i < extra else 0)

        test_indices = list(indices[start:end])
        train_indices = list(np.concatenate((indices[:start],indices[end:])))

        embeddings = self.get_embeddings(hps.embeddings)

        test_labels = [self.labels[index] for index in test_indices]
        test_data = [embeddings[index] for index in test_indices]
        test_ids = [self.ids[index] for index in test_indices]
        test_dataset = ListDataset(test_data,test_labels,test_ids)
        test_dataloader = DataLoader(test_dataset,batch_size = 1, shuffle = True)

        train_labels = [self.labels[index] for index in train_indices]
        train_data = [embeddings[index] for index in train_indices]
        train_ids = [self.ids[index] for index in train_indices]
        train_dataset = ListDataset(train_data,train_labels,train_ids)
        train_dataloader = DataLoader(train_dataset,batch_size = hps.batch_size, shuffle = True)

        return train_dataloader, test_dataloader

class ListDataset(Dataset):
    def __init__(self, data_list, labels_list,ids_list):
        self.data_list = data_list
        self.labels_list = labels_list
        self.ids_list = ids_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].squeeze()
        label = torch.tensor(self.labels_list[idx],dtype=torch.float32).squeeze()
        id = self.ids_list[idx]
        return data, label, id

@dataclass
class ManifestEntry:
    name: str
    name_internal: str
    file_path: str
    file_size: int
    sample_rate: int
    duration: float
    num_channels: int
    num_samples: int
    diagnosis: float
    diagnosis_data: list
    orthographic_path: str
    _orthographic_data: list = None
    _child_ortho_data: list = None
    _audio_data: np.ndarray = None
    _child_segments: list = None

    @classmethod
    def from_json(self, json_data: dict):
        if json_data['diagnosis'] == None:
            return
        return self(**json_data)
    
    @property
    def orthographic_data(self):
        if self._orthographic_data is None:
            self._orthographic_data = get_orthographic_data(self.orthographic_path)
        return self._orthographic_data
    
    @property
    def child_ortho_data(self):
        if self._child_ortho_data is None:
            self._child_ortho_data = [o for o in self.orthographic_data if o[-1] != '.']
        return self._child_ortho_data
    
    @property
    def audio_data(self):
        if self._audio_data is None:
            self._audio_data, sample_rate = sf.read(self.file_path)
        return self._audio_data
    
    @property
    def child_segments(self):
        if self._child_segments is None:
            audio = self.audio_data
            ortho_data = self.child_ortho_data

            indices = []
            for line in ortho_data:
                start_time = line[0]
                duration = line[1]
                end_time = start_time + duration

                start_index, end_index  = int(start_time * self.sample_rate), int(end_time * self.sample_rate)
                indices.append((start_index, end_index))
            self._child_segments = indices

            self._child_segments = []
            for start, end in indices:
                self._child_segments.append(audio[start:end])

        return self._child_segments
    
def load_data(hps):
    data = Manifest()
    filenames = [entry.name for entry in data.entries]

    embeddings = data.get_wav2vec2_embeddings()
    labels = data.labels
    if len(labels) != len(filenames):
        print('labels and filenames count mismatch')
        raise ValueError
    
    train_embs = embeddings[:-hps.num_testing]
    train_labels = labels[:-hps.num_testing]
    train_dataset = ListDataset(train_embs,train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size = hps.batch_size, shuffle = True)

    test_embs = embeddings[-hps.num_testing:]
    test_labels = labels[-hps.num_testing:]
    test_dataset = ListDataset(test_embs,test_labels)
    test_dataloader = DataLoader(test_dataset,batch_size = hps.batch_size, shuffle = True)

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    Manifest().get_egemaps_embeddings()
    