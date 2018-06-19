#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

import matplotlib
matplotlib.use('Agg')
from cnn import *
from input_construction2 import *
from generator import *
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
import pickle



[mel_spec_train, labels_train, data_info_train, data_dic_test, samples_per_speaker, too_short_records, fs, nfft, hop, trim, num_files_train, num_files_test] = load_mel_spectrogram_in_array_TIMIT('/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN/data_compression.pckl', '/scratch/paragorn/cdamhieu/datasets/TIMIT/TEST/data.pckl', ind_first_file=0, num_files_train=None, num_files_test = None, verbose=True)

"""
# Displaying spectrogram

#fig = plt.figure()
#plt.imshow(labels_train[7737:7837,400:462])
#plt.show()
#fig.savefig('/services/scratch/perception/cdamhieu/images/Speaker Identification/image_label_train_modif.png')



print("processing model ...")

# Training Parameters

"""
height_input = mel_spec_train.shape[1]
width_input = mel_spec_train.shape[2]
nb_speaker = labels_train.shape[1]

mel_spec_train = mel_spec_train.reshape(mel_spec_train.shape[0], mel_spec_train.shape[1], mel_spec_train.shape[2], 1)
#mel_spec_test = mel_spec_test.reshape(mel_spec_test.shape[0], mel_spec_test.shape[1], mel_spec_test.shape[2], 1)

batch_size = 128
steps_per_epoch = mel_spec_train.shape[0] // batch_size
epochs = 10

# Displaying array sizes
#print mel_spec_train.shape
#print labels_train.shape


model = create_cnn_model(nb_speaker = nb_speaker, height_input = height_input, width_input = width_input)
model.summary()

sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

# With validation data
#history=model1.fit(x = mel_spec_train, y = labels_train, batch_size = batch_size, epochs=epochs, verbose = 1, validation_data = (mel_spec_test, labels_test))
#model1.evaluate(mel_spec_test, labels_test)

#history = model1.fit(x = mel_spec_train, y = labels_train, batch_size = batch_size, epochs = epochs, verbose = 1)
history = model.fit_generator(generator = generator(mel_spec_train, labels_train, samples_per_speaker, nb_speaker, batch_size), steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = 1)

"""
ROOTPATH = '/services/scratch/perception/cdamhieu/weights/'
checkpointer = ModelCheckpoint(filepath=ROOTPATH+"VGG16_"+PB_FLAG+"_"+idOar+"_weights.hdf5",
                                       monitor='loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')
"""
"""
number_accurate_test = 0


for i, dic in enumerate(data_dic_test):

    
    y_predic = predict_one_record(dic, nb_speaker, model1)
    
    y = model1.predict(x = dic['mel_spectrogram'], verbose = 1)
    y_prim = y.mean(0)
    y_prim = y_prim.reshape((1,nb_speaker))
    y_predic = np.zeros([1, nb_speaker])
    y_predic[0,np.argmax(y_prim)]=1
    
    
    np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/before_mean/matrix_labels'+str(i)+'.csv', y, delimiter=",")
    np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/after_mean/matrix_labels'+str(i)+'.csv', y_predic, delimiter=",")
    np.savetxt('/services/scratch/perception/cdamhieu/results/matrix_labels/150_epochs/expected_labels/matrix_labels'+str(i)+'.csv', dic['labels'], delimiter=",")
    
    
    if np.array_equal(y_predic, dic['labels']):
        number_accurate_test += 1

percentage_accuracy = float(number_accurate_test) / len(data_dic_test)
"""
[percentage_accuracy, number_accurate_test] = predict_model_accuracy(data_dic_test, nb_speaker, model)

print percentage_accuracy
