import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import multiprocessing
import os

import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Press the green button in the gutter to run the script.
class gtzan(subset="validation"):
  def __init__(self,subset):
    self.subset=subset

    _SAMPLE_DIR = "_sample_data"
    GTZAN_DATASET_PATH = os.path.join(_SAMPLE_DIR, "gtzan")
    os.makedirs(GTZAN_DATASET_PATH, exist_ok=True)
  def _download_gtzan(self, GTZAN_DATASET_PATH):
    if os.path.exists(os.path.join(GTZAN_DATASET_PATH, "waves_gtzan.tar.gz")):
      return
    torchaudio.datasets.GTZAN(root=GTZAN_DATASET_PATH, download=True)


  GTZAN_DOWNLOAD_PROCESS = multiprocessing.Process(target=_download_gtzan)
  GTZAN_DOWNLOAD_PROCESS.start()


  def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
      axes = [axes]
    for c in range(num_channels):
      axes[c].specgram(waveform[c], Fs=sample_rate)
      if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c + 1}')
      if xlim:
        axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


  def play_audio(self,waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
      display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
      display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
      raise ValueError("Waveform with more than 2 channels are not supported.")
    

  def get_gtzan(self,GTZAN_DOWNLOAD_PROCESS,GTZAN_DATASET_PATH):
    GTZAN_DOWNLOAD_PROCESS.join()

    dataset = torchaudio.datasets.GTZAN(GTZAN_DATASET_PATH, download=True, subset=self.subset)
    return dataset
    # for i in [1, 3, 5]:
    #   waveform, sample_rate, label = dataset[i]
    #   print(len(dataset))
    #   # plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    #   play_audio(waveform, sample_rate)



