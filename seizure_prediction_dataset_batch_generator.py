#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 10:10:49 2023

@author: fabioacl
"""

import numpy as np
from tensorflow import keras

class SeizurePredictionDatasetGenerator(keras.utils.Sequence):
    """
    SeizurePredictionDatasetGenerator: generates batches of data to develop the neural networks
    
    Parameters
    ----------
    input_data : numpy.array
        Array with input samples (X).
    output_data : numpy.array
        Array with output samples (y).
    norm_values : list 
        List with mean and standard deviation of training dataset.
    batch_size : int
        Number of samples per batch.
    training_tag : str
        Tag presenting which dataset is being processed (training or validation).
        
    """
    
    def __init__(self, input_data, output_data, norm_values, batch_size, training_tag):
        self.training_tag = training_tag
        if self.training_tag=='training':
            self.input_data,self.output_data = self.balancing_dataset(input_data,output_data)
            self.number_samples = len(self.input_data)
            self.norm_mean = norm_values[0]
            self.norm_std = norm_values[1]
        else:
            self.input_data = input_data
            self.number_samples = len(self.input_data)
            self.norm_mean = norm_values[0]
            self.norm_std = norm_values[1]
            self.output_data = output_data
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(self.number_samples / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx):
        X_batch = self.input_data[idx * self.batch_size : (idx+1) * self.batch_size]
        X_batch = (X_batch-self.norm_mean)/(self.norm_std)
        y_batch = self.output_data[idx * self.batch_size : (idx+1) * self.batch_size]
        return (X_batch,y_batch)
    
    def balancing_dataset(self,inputs,labels):
        """
        Balance the dataset by oversampling the minority class
        
        Parameters
        ----------
        inputs : numpy.array
            Array with input samples (X).
        labels : numpy.array
            Array with output samples (y).

        Returns
        -------
        balanced_inputs : numpy.array
            Array with balanced input samples.
        balanced_labels : numpy.array
            Array with balanced output samples.

        """
        
        # Get negative samples indexes
        negative_samples_indexes = np.where(np.all(labels==np.array([1,0]),axis=1))[0]
        # Get number of negative samples
        length_negative_class = len(negative_samples_indexes)
        # Get positive samples indexes
        positive_samples_indexes = np.where(np.all(labels==np.array([0,1]),axis=1))[0]
        # Get number of positive samples
        length_positive_class = len(positive_samples_indexes)
    
        # Check which class contains more samples
        if length_negative_class>length_positive_class:
            majority_class_samples = negative_samples_indexes
            minority_class_samples = positive_samples_indexes
            length_minority_class = length_positive_class
        else:
            majority_class_samples = positive_samples_indexes
            minority_class_samples = negative_samples_indexes
            length_minority_class = length_negative_class
    
    
        minority_samples = inputs[minority_class_samples]
        minority_labels = labels[minority_class_samples]
        majority_samples = inputs[majority_class_samples]
        majority_labels = labels[majority_class_samples]
        
        balanced_inputs = []
        balanced_labels = []
        minority_index = 0
        
        # Oversample the minority class
        for majority_index,majority_sample in enumerate(majority_samples):
            minority_sample = minority_samples[minority_index]
            majority_label = majority_labels[majority_index]
            minority_label = minority_labels[minority_index]
            balanced_inputs.extend([majority_sample,minority_sample])
            balanced_labels.extend([majority_label,minority_label])
            minority_index = minority_index + 1
            if minority_index==length_minority_class:
                minority_index = 0
        
        balanced_inputs = np.array(balanced_inputs)
        balanced_labels = np.array(balanced_labels)
        
        return balanced_inputs,balanced_labels