#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

import numpy as np
from random import shuffle

def shuffle_list(list):

    """
    Shuffle a list of elements

    Parameters
    ----------

    list          List of elements to shuffle

    """

    shuffle(list)



def shuffle_list_10_by_10(list):

    """
    Shuffle a list of elements, ten by ten

    Parameters
    ----------
    list          List to shuffle

    Returns
    -------
    list          List shuffled
    
    """

    index_list = 0
    
    for i in range(len(list)//10):

        copy = list[index_list: index_list + 10]
        shuffle_list(copy)
        list[index_list: index_list + 10] = copy
        index_list += 10

    return list
    
    
