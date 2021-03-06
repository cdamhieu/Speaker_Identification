#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from matplotlib import pyplot as plt


def create_cnn_model(nb_speaker, height_input, width_input):

    """
    Create the model architecture of a basic CNN.

    Parameters Convolution Layers
    -----------------------------

    filters             Number of output filters in the convolution
    kernel_size         Integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    strides             Strides of the convolution along the height and width of the 2D convolution window
    activation          Activation function to use
    data_format         String, one of "channels_last" or "channels_first"
    input_shape         Shape tuple representing the input shape


    Parameters Max Pooling Layers
    -----------------------------

    pool_size           Factors by which to downscale
    strides             Strides values
    data_format         String, one of "channels_last" or "channels_first"


    Parameters Dense Layers
    -----------------------

    units               Dimensionality of the output space
    activation          Activation function to use
    

    Parameters Dropout Layer
    ------------------------

    rate                Fraction of the input unit to drop



    """
    
    # Declare Sequential model
    model = Sequential()

    # Model architecture
   
    model.add(Convolution2D(filters = 32, kernel_size=(8,1), activation='relu', input_shape=(height_input,width_input,1), data_format = "channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4,4), strides = (2,2)))
    model.add(Convolution2D(64, (8,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4,4), strides = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units = 10*nb_speaker, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(5*nb_speaker, activation='relu'))
    model.add(Dense(nb_speaker, activation='softmax'))

    return model


    
    


def create_cnn_model_VoxCeleb(nb_speaker, height_input, width_input):

    """
    Create the model architecture of the CNN.

    Parameters Convolution Layers
    -----------------------------

    filters             Number of output filters in the convolution
    kernel_size         Integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    strides             Strides of the convolution along the height and width of the 2D convolution window
    activation          Activation function to use
    data_format         String, one of "channels_last" or "channels_first"
    input_shape         Shape tuple representing the input shape


    Parameters Max Pooling Layers
    -----------------------------

    pool_size           Factors by which to downscale
    strides             Strides values
    data_format         String, one of "channels_last" or "channels_first"


    Parameters Dense Layers
    -----------------------

    units               Dimensionality of the output space
    activation          Activation function to use
    

    Parameters Dropout Layer
    ------------------------

    rate                Fraction of the input unit to drop



    """
    
    # Declare Sequential model
    model = Sequential()

    # Model architecture
   
    model.add(Convolution2D(filters = 96, kernel_size=(3,3), strides = (2,2), activation='relu', input_shape=(height_input,width_input,1), data_format = "channels_last"))
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)))
    model.add(Convolution2D(filters = 256, kernel_size = (2,2), strides = (2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)))
    model.add(Convolution2D(filters = 256, kernel_size = (2,2), strides = (1,1), activation='relu'))
    model.add(Convolution2D(filters = 256, kernel_size = (2,2), strides = (1,1), activation='relu'))
    #model.add(Convolution2D(filters = 256, kernel_size = (2,2), strides = (1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,3), strides = (3,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units = 4 * nb_speaker, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(4*nb_speaker, activation='relu'))
    model.add(Dense(nb_speaker, activation='softmax'))

    return model

        
def predict_one_record(dic, nb_speaker, model):

    """
    Predict the output for one record (already transform in one or several spectrograms).

    Parameters
    ----------
    
    dic                        Mel_spectrogram(s) of one record
    nb_speaker                 Total number of speakers in the dataset
    model                      Model of the network

    Returns
    -------

    y_predic                   Prediction of the speaker who is speaking in the record corresponding to dic

    """
    
    y = model.predict(x = dic['mel_spectrogram'], verbose = 1)
    y_prim = y.mean(0)
    y_prim = y_prim.reshape((1,nb_speaker))
    y_predic = np.zeros([1, nb_speaker])
    y_predic[0,np.argmax(y_prim)]=1

    return y_predic


def predict_model_accuracy(data_dic, nb_speaker, model):

    """
    Predict the output for a whole dictionary of data (mel-spectrogram). Calculate the percentage of well-identified speakers.

    Parameters
    ----------

    data_dic                   Dictionnaries of data to predict
    nb_speaker                 Total number of speakers in the dataset
    model                      Model of the network

    Returns
    -------

    percentage_accuracy        Percentage of well identified speakers
    number_accurate_test       Number of well identified records

    """

    number_accurate_test = 0

    for i, dic in enumerate(data_dic):
        
        y_predic = predict_one_record(dic, nb_speaker, model)

        """
        np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/before_mean/matrix_labels'+str(i)+'.csv', y, delimiter=",")
        np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/after_mean/matrix_labels'+str(i)+'.csv', y_predic, delimiter=",")
        np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/expected_labels/matrix_labels'+str(i)+'.csv', dic['labels'], delimiter=",")
        """
    
        if np.array_equal(y_predic, dic['labels']):
            number_accurate_test += 1

        percentage_accuracy = float(number_accurate_test) / len(data_dic)

    return percentage_accuracy, number_accurate_test

