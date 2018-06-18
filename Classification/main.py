#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from input_construction import *
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


#compute_data_TIMIT_train()

[mel_spec, data_info, fs, nfft, hop, trim, num_files_tot] = load_mel_spectrogram_in_array_TIMIT('/local_scratch/cdamhieu/datasets/TIMIT/TRAIN/data.pckl', ind_first_file=0, num_files=None, verbose=True)

print(mel_spec.shape)

# Make a new figure
plt.figure(figsize=(10,5))

plt.imshow(mel_spec[:,0:99])

plt.show()
