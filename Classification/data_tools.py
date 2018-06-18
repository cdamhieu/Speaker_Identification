#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Leglaive
"""


import librosa
import librosa.display
import numpy as np
import os
import pickle

def compute_STFT_data_from_file_list(wavfile_list, fs=16000, wlen_sec=0.032, hop_percent=0.5, zp_percent=0, trim=False, top_db=60, out_file=None):
    
    """
    Compute short-term Fourier transform (STFT) power and phase spectrograms from a list of wav files, 
    and save them to a pickle file.
    
    Parameters
    ----------
    
    wavfile_list                List of wav files
    fs                          Sampling rate
    wlen_sec                    STFT window length in seconds
    hop_percent                 Hop size as a percentage of the window length
    zp_percent                  Zero-padding size as a percentage of the window length
    trim                        Boolean indicating if leading and trailing silences should be trimmed
    top_db                      The threshold (in decibels) below reference to consider as silence (see librosa doc)
    out_file                   Path to the pickle file for saving the data
    
    Returns
    -------
    
    data                        A list of dictionaries, the length of the list is the same as 'wavfile_list' 
                                Each dictionary has the following fields:
                                        'file': The wav file name
                                        'power_spectrogram': The power spectrogram
                                        'phase_spectrogram': The phase spectrogram
    
    Examples
    --------
    
    fs = 16e3 # Sampling rate
    wlen_sec = 64e-3 # STFT window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    trim=False
    data_folder = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TEST'
    test_file_list = librosa.util.find_files(data_folder, ext='wav') 
    data = compute_data(test_file_list, fs=fs, wlen_sec=wlen_sec, hop_percent=hop_percent, trim=trim, zp_percent=0,             
                        out_file='test_compute_data.pckl')                   
    
    """

    # STFT parameters
    wlen = wlen_sec*fs # window length of 64 ms
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = np.int(hop_percent*wlen) # hop size
    nfft = wlen + zp_percent*wlen # number of points of the discrete Fourier transform
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window
    
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
        
        T_orig = len(x)
        x_pad = librosa.util.fix_length(x, T_orig + wlen // 2) # Padding for perfect reconstruction (see librosa doc)
        
        X = librosa.stft(x_pad, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT
        X_abs_2 = np.abs(X)**2 # Power spectrogram
        X_angle = np.angle(X)

        data[n] = {'file': file_name,
            'power_spectrogram': X_abs_2,
            'phase_spectrogram': X_angle}
        
    f = open(out_file, 'wb')
    pickle.dump([data, fs, wlen_sec, hop_percent, trim], f)
    f.close()

        
    return data


def load_STFT_data_in_array(data_file, ind_first_file=0, num_files=None, verbose=False):
    
    """
    Load the data as saved by the function 'compute_STFT_data_from_file_list' in a numpy array.
    
    Parameters
    ----------
    
    data_file                   Path to the pickle file to read
    ind_first_file              Index of the first file to be included in the resulting arrays
    num_files                   Number of files to be included in the resulting arrays
    verbose                     Boolean for verbose mode
    
    Returns
    -------
    
    data_power                  Power spectrogram as an array of size (number of frequency bins, number of time frames)
    data_phase                  Phase spectrogram as an array of size (number of frequency bins, number of time frames)
    data_info                   A list of dictionaries, the length of the list is equal to num_files
                                Each dictionary has the following fields:
                                    'file': The wav file name    
                                    'index_begin': Column index in 'data_power' and 'data_phase' where the current file begins
                                    
    fs                          Sampling rate
    wlen_sec                    STFT window length in seconds
    hop_percent                 Hop size as a percentage of the window length
    trim                        Boolean indicating if leading and trailing silences should be trimmed
    num_files                   Number of loaded files
    
    Examples
    --------
    [power_spec, phase, data_info, fs, wlen_sec, hop_percent, trim, num_files_tot] = load_data('test_compute_data.pckl',                                                                                  
                                                                                             ind_first_file=0, 
                                                                                             num_files=None, 
                                                                                             verbose=True)
    """
    
    print('loading pickle file...')
    [data_dic, fs, wlen_sec, hop_percent, trim] = pickle.load( open( data_file, "rb" ) )
    print('done\n')
    
    if num_files==None:
        num_files = len(data_dic)
    
    data_dic = data_dic[ind_first_file:ind_first_file+num_files]
    
    num_freq = data_dic[0]['power_spectrogram'].shape[0] # Number of frequency bins
    
    num_frames = 0 # Number of frames
    for n, dic in enumerate(data_dic):
        num_frames += dic['power_spectrogram'].shape[1]
                                                    
    data_info = [None] * len(data_dic)
    
    # Initialize data arrays
    data_power = np.zeros([num_freq, num_frames]) # Power spectrogram
    data_phase = np.zeros([num_freq, num_frames]) # Phase spectrogram
    
    current_ind = 0 # Current index indicating where to put the current spectrogams
    print('Loop over files')
    for n, dic in enumerate(data_dic):

        file = dic['file']
        data_info[n] = {'index_begin': current_ind, 'file': file}

        if verbose:
            print('processing file %d/%d - %s\n' % (n+1, len(data_dic), file))
        
        spectro_len = dic['power_spectrogram'].shape[1] # Number of frames of the current spectrogram
        data_power[:, current_ind:current_ind+spectro_len] = dic['power_spectrogram'] # Add to the data array
        data_phase[:, current_ind:current_ind+spectro_len] = dic['phase_spectrogram'] # Add to the data array
        
        current_ind = current_ind+spectro_len # Update the current index
        
    return data_power, data_phase, data_info, fs, wlen_sec, hop_percent, trim, num_files


def compute_STFT_data_from_file_list_TIMIT(wavfile_list, fs=16000, wlen_sec=0.032, hop_percent=0.5, zp_percent=0, trim=False, verbose=False, out_file=None):
    
    """
    Same as 'compute_STFT_data_from_file_list' function except that specific fields related to TIMIT are added to the returned and saved dictionaries.
    """

    # STFT parameters
    wlen = wlen_sec*fs # window length of 64 ms
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
    hop = np.int(hop_percent*wlen) # hop size
    nfft = wlen + zp_percent*wlen # number of points of the discrete Fourier transform
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window
    
    fs_orig = librosa.load(wavfile_list[0], sr=None)[1] # Get sampling rate    
    
    data = [None] * len(wavfile_list) # Create an empty list that will contain dictionaries
    
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
        
        T_orig = len(x)
        x_pad = librosa.util.fix_length(x, T_orig + wlen // 2) # Padding for perfect reconstruction (see librosa doc)
        
        X = librosa.stft(x_pad, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT
        X_abs_2 = np.abs(X)**2 # Power spectrogram
        X_angle = np.angle(X)

        data[n] = {'set': set_type, 'dialect': dialect, 'speaker': speaker, 'file': file_name,
            'power_spectrogram': X_abs_2,
            'phase_spectrogram': X_angle}
        
    f = open(out_file, 'wb')
    pickle.dump([data, fs, wlen_sec, hop_percent, trim], f)
    f.close()
        
    return data

def load_STFT_data_in_array_TIMIT(data_file, ind_first_file=0, num_files=None, verbose=False):
    
    """
    Same as 'load_STFT_data_in_array' function except that specific fields related to TIMIT are added to the returned dictionaries.
    """
    
    print('loading pickle file...')
    [data_dic, fs, wlen_sec, hop_percent, trim] = pickle.load( open( data_file, "rb" ) )
    print('done\n')
    
    if num_files==None:
        num_files = len(data_dic)
    
    data_dic = data_dic[ind_first_file:ind_first_file+num_files]
    
    num_freq = data_dic[0]['power_spectrogram'].shape[0] # Number of frequency bins
    
    num_frames = 0 # Number of frames
    for n, dic in enumerate(data_dic):
        num_frames += dic['power_spectrogram'].shape[1]
                                                    
    data_info = [None] * len(data_dic)
    
    # Initialize data arrays
    data_power = np.zeros([num_freq, num_frames]) # Power spectrogram
    data_phase = np.zeros([num_freq, num_frames]) # Phase spectrogram
    
    current_ind = 0 # Current index indicating where to put the current spectrogams
    print('Loop over files')
    for n, dic in enumerate(data_dic):
        
        set_type = dic['set']
        dialect = dic['dialect']
        speaker = dic['speaker']
        file = dic['file']
        data_info[n] = {'index_begin': current_ind, 'set': set_type, 'dialect': dialect, 'speaker': speaker, 
                 'file': file}

        if verbose:
            print('processing file %d/%d - %s/%s/%s/%s\n' % (n+1, len(data_dic), set_type, dialect, speaker, file))
        
        spectro_len = dic['power_spectrogram'].shape[1] # Number of frames of the current spectrogram
        data_power[:, current_ind:current_ind+spectro_len] = dic['power_spectrogram'] # Add to the data array
        data_phase[:, current_ind:current_ind+spectro_len] = dic['phase_spectrogram'] # Add to the data array
        
        current_ind = current_ind+spectro_len # Update the current index
        
    return data_power, data_phase, data_info, fs, wlen_sec, hop_percent, trim, num_files
        

def compute_duration_TIMIT(data_folder):
    """
    Compute the total amount of speech data in TIMIT train or test folders
    
    PARAMETERS
    ---------
    
    data_folder                 Path to TIMIT train or test foldes
    
    
    RETURNS
    -------
    
    total_duration              The total duration in seconds
    num_spk                     Number of speakers
    
    EXAMPLE
    --------
    
    total_duration, num_spk = compute_duration_TIMIT('/local_scratch/sileglai/datasets/clean_speech/TIMIT/TEST')
    
    """
    num_dialect = 8
    num_spk = 0
    for n in np.arange(num_dialect):
        dialect = 'DR' + str(n+1)
        spk_list = os.listdir(os.path.join(data_folder, dialect))
        num_spk += len(spk_list)
        
    
    file_list = librosa.util.find_files(data_folder, ext='wav') 
    
    total_duration = 0
    for n, file in enumerate(file_list):
        total_duration += librosa.get_duration(filename=file)
    
    return total_duration, num_spk


def compute_data_TIMIT_train(out_file=None):
    
    fs = int(16e3) # Sampling rate
    wlen_sec = 64e-3 # STFT window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    trim = True
    data_folder = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TRAIN'
    file_list = librosa.util.find_files(data_folder, ext='wav') 
            
    if out_file == None:
        out_file = os.path.join(data_folder, 'data.pckl')
        
    compute_STFT_data_from_file_list_TIMIT(file_list, 
                                            fs=fs, 
                                            wlen_sec=wlen_sec, 
                                            hop_percent=hop_percent, 
                                            zp_percent=0, 
                                            trim=trim, 
                                            verbose=True,
                                            out_file=out_file)
    
def compute_data_TIMIT_test(out_file=None):
    
    fs = int(16e3) # Sampling rate
    wlen_sec = 64e-3 # STFT window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    trim = True
    data_folder = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TEST'
    file_list = librosa.util.find_files(data_folder, ext='wav') 
    
    if out_file == None:
        out_file = os.path.join(data_folder, 'data.pckl')
        
    compute_STFT_data_from_file_list_TIMIT(file_list, 
                                            fs=fs, 
                                            wlen_sec=wlen_sec, 
                                            hop_percent=hop_percent, 
                                            zp_percent=0, 
                                            trim=trim, 
                                            verbose=True,
                                            out_file=out_file)
    
    
def write_VAE_params_to_text_file(out_file, dic_params):
    
    with open(out_file, 'w') as f:
        for key, value in dic_params.items():
            f.write('%s:%s\n' % (key, value))
            
def write_VAE_params_to_pckl_file(out_file, dic_params):
    
    f = open(out_file, 'wb')
    pickle.dump(dic_params, f)
    f.close()
            
            
#def read_VAE_params_from_text_file(in_file):    
#    dic_params = dict()
#    with open(in_file, 'r') as raw_data:
#        for item in raw_data:
#            if ':' in item:
#                key,value = item.split(':', 1)
#                dic_params[key] = value
#            else:
#                pass # deal with bad lines of text here
#    return dic_params
        
    
def compute_energy_of_TIMIT_files():
    data_folder = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TEST'
    file_list = librosa.util.find_files(data_folder, ext='wav') 
    energy = [None]*len(file_list)
    for n, wavfile in enumerate(file_list):
        x = librosa.load(wavfile, sr=None)[0]          
        energy[n] = np.sum(np.abs(x)**2)/len(x)
        print('file %d/%d' % (n, len(file_list)))
    return energy