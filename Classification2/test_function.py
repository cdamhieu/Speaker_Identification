#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from speaker import *
from save_tools import *
from input_construction2 import *
import matplotlib.pyplot as plt
import numpy as np


[mel_spec_train, labels_train, data_info_train, data_dic_test, samples_per_speaker, too_short_records, fs, nfft, hop, trim, num_files_tot] = load_mel_spectrogram_in_array_TIMIT('/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN/data_compression.pckl', '/scratch/paragorn/cdamhieu/datasets/TIMIT/TEST/data.pckl', ind_first_file=0, num_files=None, verbose=True)

save_list_too_short_records(too_short_records, out_file = '/services/scratch/perception/cdamhieu/file_list/too_short_records')
