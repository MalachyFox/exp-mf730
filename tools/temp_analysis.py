from analyse_manifest import load_manifest
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pprint import pprint

mani = load_manifest()

recording = mani[1]

pprint(recording)


data, samplerate = sf.read(recording.file_path)

figure(figsize=(10,4),dpi=100)
plt.plot(data)
plt.savefig('./tools/temp.png')