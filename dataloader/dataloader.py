import json
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from pprint import pprint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.create_manifest import get_orthographic_data
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F
import time


class Manifest:
    def __init__(self,manifest_path='/research/milsrg1/sld/exp-mf730/manifest.json'):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        manifest  = [ManifestEntry.from_json(m) for m in manifest]
        self.entries  = [m for m in manifest if m is not None]
        self.labels = [m.diagnosis for m in self.entries]
        self.manifest_path = manifest_path

    def get_wav2vec2_embeddings(self, model_name = "wav2vec2-base", embeddings_folder = '/research/milsrg1/sld/exp-mf730/embeddings'):
        embedding_file = f'{model_name}_embeddings.pt'
        path = f'{embeddings_folder}/{embedding_file}'
        if os.path.exists(path):
            print(f'\nLoading embeddings for {self.manifest_path} from {path}\n')
            embeddings = torch.load(path,weights_only=True)
        else:
            print(f'\ngenerating embeddings for {self.manifest_path} using {model_name} and saving to {embeddings_folder}\n')
            model_id = f'facebook/{model_name}'
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2Model.from_pretrained(model_id)

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
    


if __name__ == "__main__":
    Manifest().get_wav2vec2_embeddings()
    