#!/research/milsrg1/user_workspace/mf730/miniconda3/envs/myenv/

import os
from pprint import pprint
import librosa
import json
import numpy as np
import soundfile as sf

def get_filepaths(path):
    file_paths = []
    for filename in os.listdir(path):
        file_paths.append(os.path.join(path, filename))

    return file_paths

def get_mapping(path):
    mapping_dict = {}
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split()
            mapping_dict[key] = value

    return mapping_dict

def get_diagnosis_data(path):
    with open(path, 'r') as f:
        diagnosis_data = json.load(f)
    return diagnosis_data

def get_audio_metadata(file_path):
    print(f'loading {file_path}...')
    audio_data, sample_rate = sf.read(file_path)
    
    metadata = {
        "name": file_path.split('/')[-1].split('.')[0], # remove path and file extension
        "name_internal": "",
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "sample_rate": sample_rate,
        "duration": len(audio_data) / sample_rate,
        "num_channels": 1 if audio_data.ndim == 1 else audio_data.shape[0],
        "num_samples": audio_data.shape[0],
        "diagnosis": None,
        "diagnosis_data": None,
    }
    
    return metadata

def get_orthographic_data(file_path):

    orthographic_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start = parts[0]
            dur = parts[1]
            trans = ' '.join(parts[2:])
            orthographic_data.append([float(start), float(dur), trans])
    
    return orthographic_data

def collate_data(file_path: str,mapping_dict: dict,diagnosis_dict: dict,orthographic_path: str):
    metadata = get_audio_metadata(file_path)
    name = metadata['name']
    name_internal = mapping_dict[name]

    try:
        diagnosis = diagnosis_dict[name_internal]
    except:
        print(f'ERROR: no diagnosis for {name}')
        diagnosis = None
    
    metadata["name_internal"] = name_internal
    metadata["diagnosis_data"] = diagnosis
    
    if diagnosis != None:
        metadata["diagnosis"] = np.mean(diagnosis)
        metadata["orthographic_path"] = f'{orthographic_path}/{name_internal}.txt'
        #orthographic_data = get_orthographic_data(orthographic_path, name_internal)
        #metadata["orthographic_data"] = orthographic_data
        
    return metadata



if __name__ == "__main__":
    audio_data_path = "/data/milsrg1/corpora/splash/SPLSHall01/audio_16kHz/"
    mapping_data_path = "/data/milsrg1/corpora/splash/convert/lib/diagnosis/mapping.txt"
    diagnosis_data_path = "/data/milsrg1/corpora/splash/convert/lib/diagnosis/diagnosis.json"
    orthographic_path = "/home/mifs/mf730/splash/SPLSHall01/orthographic_hp/"
    output_file = '/research/milsrg1/exp-mf730/manifest.json' # current parent folder

    file_paths = get_filepaths(audio_data_path)
    mapping_dict = get_mapping(mapping_data_path)
    diagnosis_dict = get_diagnosis_data(diagnosis_data_path)

    metadatas = [collate_data(file_path,mapping_dict,diagnosis_dict,orthographic_path) for file_path in file_paths]
    metadatas = [m for m in metadatas if m != None] # remove nulls

    print(f'saved manifest to {output_file}')
    with open(output_file, 'w') as f:
        json.dump(metadatas, f, indent=4)

    