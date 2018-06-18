#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from input_construction2 import *
import matplotlib.pyplot as plt
import numpy as np

#speaker_list = speaker_list_TIMIT()
#compute_data_TIMIT_train()
"""[mel_spec_train, labels_train, data_info_train,mel_spec_test, labels_test, data_info_test, fs, nfft, hop, trim, num_files_tot] = load_mel_spectrogram_in_array_TIMIT('/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN/data.pckl',                            
                                                                                             ind_first_file=0, 
                                                                                             num_files=None, 
                                                                                             verbose=True)

print(mel_spec_train.shape)
print(mel_spec_test.shape)
print(labels_train.shape)
print(labels_test.shape)


for n in range(len(labels_train[0])):
    if n%2070 ==0:
        print(labels_train[:, n])
"""


[mel_spec_train, labels_train, data_info_train,mel_spec_test, labels_test, data_info_test, fs, nfft, hop, trim, num_files_tot] = load_mel_spectrogram_in_array_TIMIT('/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN/data.pckl',                            
                                                                                             ind_first_file=0, 
                                                                                             num_files=None, 
                                                                                             verbose=True)

print(mel_spec_train.shape)
print(labels_train.shape)
