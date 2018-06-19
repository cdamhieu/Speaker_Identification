#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

def add_sample_for_speaker(list, index_speaker, number_of_samples):

    """
    Add a number of samples for one speaker to a list containing the total number of samples per speaker.

    Parameters
    ----------

    list                            List which contains the number of samples used per speaker. 
                                    The size of the list is equals to the number of speakers, each index corresponds to one speaker.
    index_speaker                   Index of the speaker
    number_of_samples               Number of samples to add.

    Returns
    -------

    list                        List updated.

    """

    list[index_speaker] += number_of_samples

    return list


def calculate_number_samples_one_record(dic, frame_length, distance_between_frames):

    """
    Calculate the number of samples for the record corresponding to dic where dic contains the mel-spectrogram of the record.

    Parameters
    ----------

    dic                             Dictionnary which contains the informations of a record (mel-spectrogram, speaker, etc)
    frame_length                    Size of the frame on the time axis
    distance_between_frames         Distance between each frame on the time axis
    

    Returns
    -------

    number_of_samples               Number of samples used for the record corresponding to dic

    """

    number_of_samples = (dic['mel_spectrogram'].shape[1]-frame_length) // distance_between_frames + 1

    return number_of_samples

    
def recover_num_samples_without_validation(num_samples_train, num_samples_train_per_speaker, num_samples_test, data_dic, frame_length, distance_between_frames, test_sample, index_speaker = 0, too_short_records = None):

    """
    Recover the number of samples used for the training and the number of samples per speaker used for the training.
    Recover also the number of records used for the testing and a list of too short records.

    Parameters
    ----------

    num_samples_train               Number of samples used for the training
    num_samples_train_per_speaker   List which contains the number of sample per speaker.
    num_samples_test                Number of records used for testing

    data_dic                        List of dictionnaries which contain the informations of all records 
                                    (mel-spectrogram, speaker, etc)

    frame_length                    Size of the frame on the time axis
    distance_between_frames         Distance between each frame on the time axis
    test_sample                     The numero of the first sample used for testing (from 0 to 9)

    index_speaker                   Beginning index of speaker
    too_short_records               A list of too short records

    Returns
    -------

    num_samples_train               Number of samples used for the training
    num_samples_train_per_speaker   List which contains the number of sample per speaker.
    num_samples_test                Number of records used for testing

    index_speaker                   Last index of speaker
    too_short_records               A list of too short records

    """
    
    for n, dic in enumerate(data_dic):
            
        if dic['mel_spectrogram'].shape[1] >= frame_length:

            if n%10 == test_sample or n%10 == test_sample + 1:
                num_samples_test += 1

            else:
                num_samples_train += calculate_number_samples_one_record(dic, frame_length, distance_between_frames)
                num_samples_train_per_speaker = add_sample_for_speaker(num_samples_train_per_speaker, index_speaker, calculate_number_samples_one_record(dic, frame_length, distance_between_frames))
                if n%10 == 9:
                    index_speaker += 1
                    
        elif n%10 == 9 and  dic['mel_spectrogram'].shape[1] < frame_length:
            index_speaker += 1

            if too_short_records != None:
                too_short_records.append({'dataset': "TRAIN",
                                          'set_type': dic['set'],
                                          'dialect': dic['dialect'],
                                          'speaker': dic['speaker'],
                                          'file': dic['file']})

        else:

            if too_short_records != None:
                too_short_records.append({'dataset': "TRAIN",
                                          'set_type': dic['set'],
                                          'dialect': dic['dialect'],
                                          'speaker': dic['speaker'],
                                          'file': dic['file']})
        
    return num_samples_train, num_samples_train_per_speaker, num_samples_test, index_speaker, too_short_records
    

def recover_frames_spec_one_record(num_samples, mel_specs, dic, frame_length, distance_between_frames, current_ind = 0):

    current_ind_mel_dic = 0
    
    for i in range(num_samples):

        mel_specs[current_ind, :, :] = dic['mel_spectrogram'][:, current_ind_mel : current_ind_mel + frame_length]
        current_ind += 1
        current_ind_mel += distance_between_frames

    return mel_specs, current_ind


