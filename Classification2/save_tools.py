#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

def save_list_of_dictionnaries(list_to_save, out_file = None):

    data_folder = '/services/scratch/perception/cdamhieu/file_list/'

    if out_file == None:
        out_file = os.path.join(data_folder, 'list.txt')

    saved_file = open(out_file, "w")

    for i in range(len(list_to_save)):

        saved_file.write("numero : " + str(list_to_save[i]['numero']) + "   name : " + list_to_save[i]['name'] + "\n")

    saved_file.closed()
        
    return saved_file
        

def save_list(list_to_save, out_file = None):

    data_folder = '/services/scratch/perception/cdamhieu/file_list/'

    if out_file == None:
        out_file = os.path.join(data_folder, 'list.txt')

    saved_file = open(out_file, "w")

    for i in range(len(list_to_save)):

        saved_file.write("speaker : " + str(i) + "   samples per speaker : " + str(list_to_save[i]) + "\n")

    saved_file.close()
    
    return saved_file
        

def save_list_too_short_records(list_to_save, out_file = None):

    data_folder = '/services/scratch/perception/cdamhieu/file_list/'

    if out_file == None:
        out_file = os.path.join(data_folder, 'list.txt')

    saved_file = open(out_file, "w")

    for i in range(len(list_to_save)):

        saved_file.write("dataset : " + list_to_save[i]['dataset'] + "\n")
        saved_file.write("set_type : " + list_to_save[i]['set_type'] + "\n")
        saved_file.write("dialect : " + list_to_save[i]['dialect'] + "\n")
        saved_file.write("speaker : " + list_to_save[i]['speaker'] + "\n")
        saved_file.write("file : " + list_to_save[i]['file'] + "\n")

    saved_file.close()
    
    return saved_file

        
