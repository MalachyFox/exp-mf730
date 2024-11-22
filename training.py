from dataloader.dataloader import Manifest
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch

data = Manifest()

embeddings = data.get_wav2vec2_embeddings()
labels = data.labels

if len(embeddings) != len(labels):
    raise ValueError