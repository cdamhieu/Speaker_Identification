#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import librosa
import librosa.display
import numpy as np
import os
import pickle


def compute_mel_spectrogram_from_file_list(wavfile_list, fs=16000, nfft=1024, hop=160, zp_percent=0, trim=False, top_db=60, verbose=False, out_file=None):
    
    """
    Compute mel-spectrograms from a list of wav files, 
    and save them to a pickle file.
    
    Parameters
    ----------
    
    wavfile_list                List of wav files
    fs                          Sampling rate
    nfft                        Length of the FFT window
    hop                         Number of samples between successive frames

    trim                        Boolean indicating if leading and trailing silences should be trimmed
    top_db                      The threshold (in decibels) below reference to consider as silence (see librosa doc)
    out_file                    Path to the pickle file for saving the data
    
    Returns
    -------
    
    data                        A list of dictionaries, the length of the list is the same as 'wavfile_list' 
                                Each dictionary has the following fields:
                                        'file': The wav file name
                                        'mel_spectrogram': The mel-spectrogram
    
    Examples
    --------
    
    fs = 16e3 # Sampling rate
    nfft = 1024
    hop = 160
    
    trim=False
    data_folder = '/local_scratch/cdamhieu/datasets/TIMIT/TEST'
    test_file_list = librosa.util.find_files(data_folder, ext='wav') 
    data = compute_data(test_file_list, fs=fs, wlen_sec=wlen_sec, hop_percent=hop_percent, trim=trim, zp_percent=0,             
                        out_file='test_compute_data.pckl')                   
    
    """



    # Mel-spectrogram parameters
    
    #win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window
    
    fs_orig = librosa.load(wavfile_list[0], sr=None)[1] # Get sampling rate    

    data = [None] * len(wavfile_list) # Create an empty list that will contain dictionaries

    
    for n, wavfile in enumerate(wavfile_list):
        
        path, file_name = os.path.split(wavfile)
        
        if fs==fs_orig: 
            x = librosa.load(wavfile, sr=None)[0] # Load wav file without resampling                      
        else:
            print('resampling while loading with librosa')
            x = librosa.load(wavfile, sr=fs)[0] # Load wav file with resampling                       
        
        if trim:
            x = librosa.effects.trim(x, top_db=top_db)[0] # Trim leading and trailing silences
        
        #T_orig = len(x)
        #x_pad = librosa.util.fix_length(x, T_orig + wlen // 2) # Padding for perfect reconstruction (see librosa doc)
        
        X = librosa.feature.melspectrogram(x, n_fft=nfft, hop_length=hop) # Mel-spectrogram
        log_X = np.log(1+10000*X)

        data[n] = {'file': file_name,
            'mel_spectrogram': log_X}
        
    f = open(out_file, 'wb')
    pickle.dump([data, fs, nfft, hop, trim], f)
    f.close()
        
    return data



def compute_mel_spectrogram_from_file_list_TIMIT(wavfile_list, fs=16000, nfft=1024, hop=160, zp_percent=0, trim=False, top_db=60, verbose=False, out_file=None):
    
    """
    Same as 'compute_mel_spectrogram_from_file_list' function except that specific fields related to TIMIT are added to the returned and saved dictionaries.
    """

    # mel-spectrogram parameters
    
    fs_orig = librosa.load(wavfile_list[0], sr=None)[1] # Get sampling rate    
    
    speaker_list = speaker_list_TIMIT()
    data = [None] * len(wavfile_list) # Create an empty list that will contain dictionaries
    index_speaker = 0 # Initialization of the index speaker
    
    for n, wavfile in enumerate(wavfile_list):
        
        path, file_name = os.path.split(wavfile)
        path, speaker = os.path.split(path)
        path, dialect = os.path.split(path)
        path, set_type = os.path.split(path)
        
        if verbose:
            print('processing %s/%s/%s/%s\n' % (set_type, dialect, speaker, file_name))
        
        if fs==fs_orig: 
            x = librosa.load(wavfile, sr=None)[0] # Load wav file without resampling                      
        else:
            print('resampling while loading with librosa')
            x = librosa.load(wavfile, sr=fs)[0] # Load wav file with resampling                       
        
        if trim:
            with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4]+'.PHN'), 'r') as f:
                first_line = f.readline() # Read the first line
                for last_line in f: # Loop through the whole file reading it all
                    pass      
                
            if not('#' in first_line) or not('#' in last_line):
                raise NameError('The first or last lines of the .phn file should contain #')
            
            ind_beg = int(first_line.split(' ')[1])
            ind_end = int(last_line.split(' ')[0])
            x = x[ind_beg:ind_end]
        
        #T_orig = len(x)
        #x_pad = librosa.util.fix_length(x, T_orig + wlen // 2) # Padding for perfect reconstruction (see librosa doc)

        X = librosa.feature.melspectrogram(x, n_fft=nfft, hop_length=hop) # Mel-spectrogram        
        log_X = np.log(1+10000*X) # Compression

        X_length = len(X[0]) # Horizontal size of the mel-spectrogram
        output = np.zeros([len(speaker_list), X_length]) # Create an array of zeros of the size of the mel-spectrogram


        for i, speaker_l in enumerate(speaker_list):
            if speaker == speaker_l: # Comparison between the current speaker and the i-th speaker of the speaker list
                index_speaker = i
        
        output[index_speaker]=np.ones([1, X_length])
                
        data[n] = {'set': set_type, 'dialect': dialect, 'speaker': speaker, 'file': file_name, 'label': output,
            'mel_spectrogram': log_X}
        
    f = open(out_file, 'wb')
    pickle.dump([data, fs, nfft, hop, trim], f)
    f.close()
        
    return data




def load_mel_spectrogram_in_array_TIMIT(data_file, ind_first_file=0, num_files=None, verbose=False):
    
    """
    Load the data as saved by the function 'compute_mel_spectrogram_from_file_list_TIMIT' in a numpy array.
    
    Parameters
    ----------
    
    data_file                   Path to the pickle file to read
    ind_first_file              Index of the first file to be included in the resulting arrays
    num_files                   Number of files to be included in the resulting arrays
    verbose                     Boolean for verbose mode
    
    Returns
    -------
    
    data_mel_train              Mel-spectrogram of the train data as an array of size (number of frequency bins, number of time frames)
    data_label_train            An array corresponding of the expected outputs of the CNN for the test data
    data_info_train             A list of dictionaries of the train data, the length of the list is equal to num_files
                                Each dictionary has the following fields:
                                    'file': The wav file name    
                                    'index_begin': Column index in 'data_mel_train' and 'data_label_train' where the current file begins

    data_mel_test               Mel-spectrogram of the test data as an array of size (number of frequency bins, number of time frames)
    data_label_test             An array corresponding of the expected outputs of the CNN for the train data
    data_info_test              A list of dictionaries of the test data, the length of the list is equal to num_files
                                Each dictionary has the following fields:
                                    'file': The wav file name    
                                    'index_begin': Column index in 'data_mel_test' and 'data_label_test' where the current file begins
                                    
    fs                          Sampling rate
    nfft                        Length of the FFT window
    hop                         Number of samples between successive frames
    trim                        Boolean indicating if leading and trailing silences should be trimmed
    num_files                   Number of loaded files
    
    Examples
    --------
    [mel_spec_train, labels_train,  data_info_train, mel_spec_test, labels_test, data_info_test, fs, nfft, hop, trim, num_files_tot] = load_data('test_compute_data.pckl',                                                                                  
                                                                                             ind_first_file=0, 
                                                                                             num_files=None, 
                                                                                             verbose=True)
    """

    
    print('loading pickle file...')
    [data_dic, fs, nfft, hop, trim] = pickle.load( open( data_file, "rb" ) )
    print('done\n')
    
    if num_files==None:
        num_files = len(data_dic)
    
    data_dic = data_dic[ind_first_file:ind_first_file+num_files]

    
    num_freq = data_dic[0]['mel_spectrogram'].shape[0] # Number of frequency bins
    
    num_frames_train = 0 # Number of frames for the train
    num_frames_test = 0 # Number of frames for the test
    
    for n, dic in enumerate(data_dic):
        
        if n%8 == 0 or n%9 == 0:
            num_frames_test += dic['mel_spectrogram'].shape[1]
            
        else:
            num_frames_train += dic['mel_spectrogram'].shape[1]

    num_speakers = data_dic[0]['label'].shape[0] # Number of speakers
                                                    
    data_info_train = [None]
    data_info_test = [None]
    
    
    # Initialize data arrays
    data_mel_train = np.zeros([num_freq, num_frames_train]) # Mel-spectrogram for the train data
    data_label_train = np.zeros([num_speakers, num_frames_train]) # Expected output for the train data
    data_mel_test = np.zeros([num_freq, num_frames_test]) # Mel-spectrogram for the test data
    data_label_test = np.zeros([num_speakers, num_frames_test]) # Expected output for the test data

    
    current_ind_train = 0 # Current index indicating where to put the current spectrogams for the train data
    current_ind_test = 0 # Current index indicating where to put the current spectrogams for the test data

    
    print('Loop over files')
    for n, dic in enumerate(data_dic):
        
        set_type = dic['set']
        dialect = dic['dialect']
        speaker = dic['speaker']
        file = dic['file']


        if verbose:
            print('processing file %d/%d - %s/%s/%s/%s\n' % (n+1, len(data_dic), set_type, dialect, speaker, file))
        
        spectro_len = dic['mel_spectrogram'].shape[1] # Number of frames of the current spectrogram

        if n%8 == 0 or n%9 == 0:
        
            data_mel_test[:, current_ind_test:current_ind_test+spectro_len] = dic['mel_spectrogram'] # Add to the data array
            data_label_test[:,current_ind_test:current_ind_test+spectro_len] = dic['label']
        
            current_ind_test = current_ind_test+spectro_len # Update the current index
            data_info_test.append({'index_begin': current_ind_test, 'set': set_type, 'dialect': dialect, 'speaker': speaker,
                 'file': file})
        else:
            
            data_mel_train[:, current_ind_train:current_ind_train+spectro_len] = dic['mel_spectrogram'] # Add to the data array
            data_label_train[:,current_ind_train:current_ind_train+spectro_len] = dic['label']
        
            data_info_train.append({'index_begin': current_ind_train, 'set': set_type, 'dialect': dialect, 'speaker': speaker,
                 'file': file})
            
    return data_mel_train, data_label_train, data_info_train, data_mel_test, data_label_test, data_info_test, fs, nfft, hop, trim, num_files



def compute_data_TIMIT_train(out_file=None):

    """
    wavfile_list                List of wav files
    fs                          Sampling rate
    nfft                        Length of the FFT window
    hop                         Number of samples between successive frames

    """
    
    fs = int(16e3) # Sampling rate
    hop = 160  # hop size as a percentage of the window length
    trim = True
    data_folder = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN'
    file_list = librosa.util.find_files(data_folder, ext='wav') 
            
    if out_file == None:
        out_file = os.path.join(data_folder, 'data.pckl')
        
    compute_mel_spectrogram_from_file_list_TIMIT(file_list, 
                                            fs=fs,
                                            nfft=1024,
                                            hop=hop,
                                            trim=trim, 
                                            verbose=True,
                                            out_file=out_file)


def compute_data_TIMIT_test(out_file=None):

    """
    wavfile_list                List of wav files
    fs                          Sampling rate
    nfft                        Length of the FFT window
    hop                         Number of samples between successive frames

    """
    
    fs = int(16e3) # Sampling rate
    hop = 160  # hop size as a percentage of the window length
    trim = True
    data_folder = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TEST'
    file_list = librosa.util.find_files(data_folder, ext='wav') 
            
    if out_file == None:
        out_file = os.path.join(data_folder, 'data.pckl')
        
    compute_mel_spectrogram_from_file_list_TIMIT(file_list, 
                                            fs=fs,
                                            nfft=1024,
                                            hop=hop,
                                            trim=trim, 
                                            verbose=True,
                                            out_file=out_file)



def speaker_list_TIMIT():

    """
    Return a list of all speakers of TIMIT_TRAIN

    Returns
    -------

    speaker_list                List of all speakers in the TIMIT TRAIN

    """

    speaker_list = []
    data_folder = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN'

    for i in range(1,8):
        strg = "DR" + str(i)
        path = os.path.join(data_folder, strg)
        speaker_list.extend(os.listdir(path))

    return speaker_list
        
