#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

import os



def speaker_list_TIMIT_train():

    """
    Return the list of all speakers of TIMIT_TRAIN

    Returns
    -------

    speaker_list                List of all speakers in the TIMIT TRAIN

    """

    speaker_list = []
    data_folder = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN'

    for i in range(1,9):
        strg = "DR" + str(i)
        path = os.path.join(data_folder, strg)
        speaker_list.extend(os.listdir(path))

    speaker_dic = [None] * len(speaker_list)
    
    for n, speaker in enumerate(speaker_list):
        speaker_dic[n] = {'numero':n, 'name':speaker}
    
    return speaker_dic
  


def speaker_list_TIMIT_test():

    """
    Return the list of all speakers of TIMIT_TEST

    Returns
    -------

    speaker_list                List of all speakers in the TIMIT TEST

    """

    speaker_list = []
    data_folder = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TEST'

    for i in range(1,9):
        strg = "DR" + str(i)
        path = os.path.join(data_folder, strg)
        speaker_list.extend(os.listdir(path))

    speaker_dic = [None] * len(speaker_list)
    
    for n, speaker in enumerate(speaker_list):
        speaker_dic[n] = {'numero':n, 'name':speaker}
    
    return speaker_dic




def speaker_list_TIMIT():

    """
    Return the list of all speakers of TIMIT_TEST

    Returns
    -------

    speaker_list                List of all speakers in the TIMIT TRAIN

    """

    speaker_list = []
    
    data_folder_train = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TRAIN'
    data_folder_test = '/scratch/paragorn/cdamhieu/datasets/TIMIT/TEST'

    for i in range(1,9):
        strg = "DR" + str(i)
        path = os.path.join(data_folder_train, strg)
        speaker_list.extend(os.listdir(path))

    for i in range(1,9):
        strg = "DR" + str(i)
        path = os.path.join(data_folder_test, strg)
        speaker_list.extend(os.listdir(path))

    print("len(speaker_dic) " + str(len(speaker_list)))
    speaker_dic = [None] * len(speaker_list)
    
    for n, speaker in enumerate(speaker_list):
        speaker_dic[n] = {'numero':n, 'name':speaker}
    
    return speaker_dic
