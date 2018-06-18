#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

from random import randint
import numpy as np


def generator(mel_spectrogram, labels, samples_per_speaker, nb_speaker, batch_size):

    while True:
        
        for i in range((mel_spectrogram.shape[0] + batch_size - 1) // batch_size):

            X = np.zeros([batch_size, mel_spectrogram.shape[1], mel_spectrogram.shape[2], mel_spectrogram.shape[3]])
            y = np.zeros([batch_size, nb_speaker])

            for j in range(batch_size):

                index_speaker = randint(0, nb_speaker-1)
                begin_index = 0

                for k in range(index_speaker):
                    begin_index += samples_per_speaker[k]

                index = randint(begin_index, begin_index + samples_per_speaker[index_speaker] - 1)

                X[j,:,:,:] = mel_spectrogram[index,:,:,:]
                y[j,:] = labels[index,:]

            yield X, y
                            
    
