#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@author: Caroline Dam Hieu
"""

import matplotlib.pyplot as plt


"""

Plot the loss and accuracy curves for training and validation

"""


def loss_curve_training_validation(history):

    """
    Plot the loss curves for both training and validation (to use if we have validation data) and save it in a file .png

    Parameters
    ----------
    history            Model fitted

    """
    
    fig = plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    fig.savefig('/services/scratch/perception/cdamhieu/results/curves/accuracy_loss_per_epoch/loss_curves_training_validation.png')


def loss_curve_training(history):

    """
    Plot the loss curves for training only (no validation data used) and save it in a file .png

    Parameters
    ----------
    history            Model fitted

    """
    
    fig = plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.legend(['Training loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    fig.savefig('/services/scratch/perception/cdamhieu/results/curves/accuracy_loss_per_epoch/loss_curves_training.png')




def accuracy_curve_training_validation(history):

    """
    Plot the accuracy curves for both training and validation (to use if we have validation data) and save it in a file .png

    Parameters
    ----------
    history            Model fitted

    """

    fig = plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    fig.savefig('/services/scratch/perception/cdamhieu/results/curves/accuracy_loss_per_epoch/accuracy_curves_training_validation.png')

    

def accuracy_curve_training(history):

    """
    Plot the accuracy curves for training only (no validation data used) and save it in a file .png

    Parameters
    ----------
    history            Model fitted

    """
 
    fig = plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.legend(['Training Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    fig.savefig('/services/scratch/perception/cdamhieu/results/curves/accuracy_loss_per_epoch/accuracy_curves_training.png')

