#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:37:49 2021

@author: fabioacl
"""

#%% Import Libraries
import numpy as np
import math
import random
from scipy.special import comb
import os
import re
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import pandas as pd
import pickle
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import *
from scipy import signal
from scipy.signal import butter,filtfilt,iirnotch
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import keras.backend as K
from tensorflow import keras
from Evaluation import Evaluation

#%% Classes
''' Batch Generator '''

class MyCustomGenerator(keras.utils.Sequence):
  
  def __init__(self, input_data, output_data, norm_values, batch_size, training_tag) :
    if training_tag=='training':
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
    
  def balancing_dataset(self,dataset,labels):
    # Get the number of samples in the dataset
    dataset_size = len(labels)
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
        length_majority_class = length_negative_class
        minority_class_samples = positive_samples_indexes
        length_minority_class = length_positive_class
    else:
        majority_class_samples = positive_samples_indexes
        length_majority_class = length_positive_class
        minority_class_samples = negative_samples_indexes
        length_minority_class = length_negative_class
    
    
    minority_samples = dataset[minority_class_samples]
    minority_labels = labels[minority_class_samples]
    majority_samples = dataset[majority_class_samples]
    majority_labels = labels[majority_class_samples]
    
    balanced_dataset = []
    balanced_labels = []
    minority_index = 0
    
    for majority_index,majority_sample in enumerate(majority_samples):
        minority_sample = minority_samples[minority_index]
        majority_label = majority_labels[majority_index]
        minority_label = minority_labels[minority_index]
        balanced_dataset.extend([majority_sample,minority_sample])
        balanced_labels.extend([majority_label,minority_label])
        minority_index = minority_index + 1
        if minority_index==length_minority_class:
            minority_index = 0
    
    return np.array(balanced_dataset),np.array(balanced_labels)

#%% Functions

def filtering_signals(data,fs,low_freq,high_freq,notch_freq,order):
    low_wn = low_freq/(0.5*fs)
    high_wn = high_freq/(0.5*fs)
    b,a = butter(order,low_wn,'low')
    data = filtfilt(b,a,data,axis=1)
    b,a = butter(order,high_wn,'high')
    data = filtfilt(b,a,data,axis=1)
    b,a = iirnotch(notch_freq,35,fs)
    data = filtfilt(b,a,data,axis=1)
    
    return data

def load_file(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

'''Get interictal and preictal data. Seizure Prediction Horizon (SPH) represents the interval that
the patient has to prepare himself/herself to the seizure, i.e., if a SPH of 10 minutes is used,
the seizure only happens at least 10 minutes after the seizure alarm. Seizure Occurence Period (SOP) 
represents the interval when the seizure occurs. For predicting the seizure, a preictal interval 
equals to the SOP is used. Therefore, it is expected that the seizure occurs after the sph and
inside the following sop interval. For example, if a SOP interval of 40 minutes is used 
it is expected that the seizure occurs inside the 40 minutes after the SPH.'''
def get_dataset_labels(datetimes,sop,sph):
    num_seizures = len(datetimes)
    sop_datetime = pd.Timedelta(minutes=sop)
    sph_datetime = pd.Timedelta(minutes=sph)
    interictal_begin_indexes = [0]*num_seizures
    interictal_end_indexes = []
    preictal_begin_indexes = []
    preictal_end_indexes = []
    dataset_labels = []
    
    for seizure_index,seizure_datetimes in enumerate(datetimes):
        # Get last seizure datetime
        last_window_seizure_datetime = pd.to_datetime(seizure_datetimes[-1][-1],unit='s')
        # Get beginning of pre-ictal period
        begin_sop_seizure_datetime = last_window_seizure_datetime - sop_datetime - sph_datetime
        # Get last datetime of SOP
        last_sop_seizure_datetime = begin_sop_seizure_datetime + sop_datetime
        inside_seizure_preictal = False
        for datetime_index,posix_datetime in enumerate(seizure_datetimes):
            if inside_seizure_preictal==False:
                # Get last window datetime
                last_sample_datetime = pd.to_datetime(posix_datetime[-1],unit='s')
                # If the last window datetime is inside SOP, inter-ictal ends and pre-ictal begins
                if begin_sop_seizure_datetime<last_sample_datetime:
                    interictal_end_indexes.append(datetime_index-1)
                    preictal_begin_indexes.append(datetime_index)
                    inside_seizure_preictal = True
        preictal_end_indexes.append(datetime_index)

    for seizure_index in range(num_seizures):
        seizure_interictal_begin = interictal_begin_indexes[seizure_index]
        seizure_interictal_end = interictal_end_indexes[seizure_index]
        seizure_preictal_begin = preictal_begin_indexes[seizure_index]
        seizure_preictal_end = preictal_end_indexes[seizure_index]
        seizure_sph_begin = preictal_end_indexes[seizure_index]+1
        seizure_sph_end = len(datetimes[seizure_index])
        seizure_length_interictal = seizure_interictal_end - seizure_interictal_begin + 1
        seizure_length_preictal = seizure_preictal_end - seizure_preictal_begin + 1
        seizure_length_sph = seizure_sph_end - seizure_sph_begin
        # Inter-ictal labels (0)
        seizure_interictal_labels = np.zeros((seizure_length_interictal,))
        # Pre-ictal labels (1)
        seizure_preictal_labels = np.ones((seizure_length_preictal,))
        # SPH labels (0)
        seizure_sph_labels = np.zeros((seizure_length_sph,))
        seizure_dataset_labels = np.concatenate((seizure_interictal_labels,seizure_preictal_labels,seizure_sph_labels))
        
        dataset_labels.append(seizure_dataset_labels)
    
    return dataset_labels

def get_training_dataset(dataset,labels,datetimes,seizure_onset_datetimes,training_time,validation_ratio,sph,num_training_seizures):
    interictal_time = pd.Timedelta(hours = training_time)
    training_dataset = dataset[:num_training_seizures]
    training_labels = labels[:num_training_seizures]
    training_datetimes = datetimes[:num_training_seizures]
    training_time = pd.Timedelta(hours = training_time)
    sph_datetime = pd.Timedelta(minutes=sph)
    
    for seizure_index,seizure_datetimes in enumerate(training_datetimes):
        # Get last seizure datetime
        begin_seizure_datetime = seizure_onset_datetimes[seizure_index]
        # Get beginning of SPH (Last datetime minus SPH duration)
        begin_seizure_sph_datetime = begin_seizure_datetime - sph_datetime
        # Get beginning of training time (Last datetime minus Training data duration)
        begin_interictal_training_seizure_datetime = begin_seizure_datetime - training_time
        training_indexes = []
        for datetime_index,posix_datetime in enumerate(seizure_datetimes):
            # Get last window datetime
            last_sample_datetime = pd.to_datetime(posix_datetime[-1],unit='s')
            # If the last window datetime is inside the training period (last 4 hours and before SPH) it is considered
            if last_sample_datetime>begin_interictal_training_seizure_datetime and last_sample_datetime<begin_seizure_sph_datetime:
                training_indexes.append(datetime_index)
        training_indexes = np.array(training_indexes)
        training_dataset[seizure_index] = training_dataset[seizure_index][training_indexes]
        training_labels[seizure_index] = training_labels[seizure_index][training_indexes]
        training_datetimes[seizure_index] = training_datetimes[seizure_index][training_indexes]
    
    return training_dataset,training_labels,training_datetimes

def merge_seizure_datasets(dataset,labels):
    num_seizures = len(dataset)
    merged_dataset = dataset[0]
    merged_labels = labels[0]
    
    for seizure_index in range(1,num_seizures):
        merged_dataset = np.concatenate((merged_dataset,dataset[seizure_index]))
        merged_labels = np.concatenate((merged_labels,labels[seizure_index]))
    
    return merged_dataset,merged_labels

''' The temporal correction consists in introducing 0s in gaps. For each window gap, a 0 is introduced.
    For example, if a gap is longer than 10 seconds, a 0 is introduced in labels array; if a gap is longer than 20 seconds, 
    two 0s are introduced; and so on.'''
def temporal_firing_power(y_pred,datetimes,sop,sph,window_seconds,threshold):
    sample_freq = 256
    num_samples = len(y_pred)
    sop_samples = sop*60/window_seconds
    fp_step = 1/sop_samples
    sop_time = pd.Timedelta(minutes=sop)
    # Cumulative Values
    firing_power_windows = [y_pred[0]*fp_step]
    # Firing power value for each timestep
    firing_power_values = [y_pred[0]*fp_step]
    # Last datetime of the first window
    last_previous_sample_posix_datetime = datetimes[0][-1]
    last_previous_sample_datetime = pd.to_datetime(last_previous_sample_posix_datetime,unit='s')
    firing_power_datetimes = np.array([last_previous_sample_datetime])
    firing_power_reset = False
    
    for sample_index in range(1,num_samples):
        sample_label = y_pred[sample_index]
        # Last datetime of the current window
        last_current_sample_posix_datetime = datetimes[sample_index][-1]
        last_current_sample_datetime = pd.to_datetime(last_current_sample_posix_datetime,unit='s')
        # Difference between the current window and the previous window to verify whether there is a gap
        diff_windows = last_current_sample_datetime - last_previous_sample_datetime
        diff_windows_seconds = diff_windows.total_seconds()
        # If the window contains data from the previous window, the firing power step is not totally considered. For example
        # if a window contains 6.5 seconds of new data only 65% of the new firiing power step will be summed to the others.
        coeff_step = diff_windows_seconds/window_seconds
        if coeff_step<=1:
            step_value = sample_label*fp_step*coeff_step
        elif coeff_step>1:
            coeff_gap = coeff_step - 1 # 1 is the coeff step if everything is ok
            step_value = sample_label*fp_step
            gap_penalty = fp_step*(coeff_step-1)
            # If there is a gap between the two consecutive windows, the algorithm gives a penalty to the firing power score. In this case
            # the algorithm subtracts a value equivalent to the gap duration. For example, if the difference between the two windows
            # is 15 seconds, 5 of the 15 seconds are considered as a gap. Therefore, the new step will have a gap penalty of 50% of
            # a normal firing power positive step (class 1).
            step_value = step_value - gap_penalty
                
        firing_power_values = np.append(firing_power_values,step_value)
        firing_power_datetimes = np.append(firing_power_datetimes,last_current_sample_datetime)
        last_firing_power_window_datetime = firing_power_datetimes[-1] - sop_time
        
        # Remove elements that are outside the firing power window
        remove_indexes = np.where(firing_power_datetimes<last_firing_power_window_datetime)[0]
        
        firing_power_datetimes = np.delete(firing_power_datetimes,remove_indexes)
        firing_power_values = np.delete(firing_power_values,remove_indexes)
        
        firing_power_window_value = sum(firing_power_values)
        
        if firing_power_window_value<0:
            firing_power_values = [0]
            firing_power_datetimes = [last_current_sample_datetime]
            firing_power_window_value = 0

        firing_power_windows.append(firing_power_window_value)
        
        last_previous_sample_datetime = last_current_sample_datetime
    
    firing_power_windows = np.array(firing_power_windows)
    # Convert the firing power scores in classes
    filtered_y_pred = np.where(firing_power_windows >= threshold, 1, 0)
    
    inside_refractory_time = False
    # When there is an alarm, there cannot be another while it is under the refractory time (SOP+SPH).
    # This is performed because we there is an alarm, the patient will have a SPH to prepare himself for a seizure and a SOP
    # when the seizure occurs.
    refractory_time_duration = pd.Timedelta(minutes=sop+sph)
    for sample_index in range(1,num_samples):
        current_label = filtered_y_pred[sample_index]
        current_datetime = datetimes[sample_index][-1]
        current_datetime = pd.to_datetime(current_datetime,unit='s')
        if current_label==1 and inside_refractory_time==False:
            end_refractory_time = current_datetime + refractory_time_duration
            inside_refractory_time = True
        elif current_label==1 and inside_refractory_time:
            filtered_y_pred[sample_index] = 0
        if inside_refractory_time:
            if current_datetime>end_refractory_time:
                inside_refractory_time = False
    return firing_power_windows,filtered_y_pred

'''Function to evaluate the model. The data do not contain the sph.'''
def evaluate_model(y_pred,y_true,datetimes,sop,sph,seizure_onset_datetime):
    sop_time = pd.Timedelta(minutes=sop)
    sph_time = pd.Timedelta(minutes=sph)
    window_datetime_step = pd.Timedelta(nanoseconds=1e9/256)
    first_datetime = pd.to_datetime(datetimes[0][0],unit='s')
    last_datetime = seizure_onset_datetime - sph_time
    begin_sop_datetime = last_datetime - sop_time
    refractory_time = sop_time + sph_time
    inside_refractory_time = False
    inside_sop_time = False
    possible_firing_time = 0
    
    true_alarms = 0
    false_alarms = 0
    
    alarm_indexes = np.where(y_pred==1)[0]
    
    num_windows = len(datetimes)
    
    # Get begin and end datetimes from the first window
    last_window_begin_datetime = pd.to_datetime(datetimes[0][0],unit='s')
    last_window_end_datetime = pd.to_datetime(datetimes[0][-1],unit='s')
    
    # Just to initialise datetime
    finish_refractory_time_datetime = last_window_begin_datetime
    # Get first window time length
    last_window_duration = (last_window_end_datetime - last_window_begin_datetime + window_datetime_step).seconds
    possible_firing_time = last_window_duration
    
    for window_index in range(1,num_windows):
        window_datetimes = datetimes[window_index]
        current_window_begin_datetime = pd.to_datetime(window_datetimes[0],unit='s')
        current_window_end_datetime = pd.to_datetime(window_datetimes[-1],unit='s')
        
        if current_window_begin_datetime < last_window_end_datetime:
            current_window_begin_datetime = last_window_end_datetime
        
        if window_index in alarm_indexes:
            inside_refractory_time = True
            finish_refractory_time_datetime = current_window_begin_datetime + refractory_time
        
        if current_window_end_datetime > finish_refractory_time_datetime:
            inside_refractory_time = False
        
        if current_window_end_datetime > begin_sop_datetime:
            inside_sop_time = True
        
        if inside_refractory_time == False and inside_sop_time==False:
            current_window_duration = (current_window_end_datetime - current_window_begin_datetime + window_datetime_step).seconds
            possible_firing_time += current_window_duration
        
        last_window_begin_datetime = current_window_begin_datetime
        last_window_end_datetime = current_window_end_datetime
    
    possible_firing_time /= 3600 # Convert from seconds to hours
    
    for alarm_index in alarm_indexes:
        predicted_label = y_pred[alarm_index]
        true_label = y_true[alarm_index]
        if predicted_label==true_label and predicted_label==1:
            true_alarms += 1
        elif predicted_label!=true_label and predicted_label==1:
            false_alarms += 1
    
    sensitivity = true_alarms
    # To compute the FPR/h we must remove the time when the alarm cannot be fired (refractory time).
    fpr_h = false_alarms/possible_firing_time
    
    return sensitivity,fpr_h
    
def save_results(patient_number,filename,all_sensitivities,all_fpr_h,sop,tested_seizures,ss_rand_pred,ss_surrogate,fpr_h_surrogate):
    avg_ss = np.mean(all_sensitivities)
    avg_fpr_h = np.mean(all_fpr_h)
    beat_rp = avg_ss>ss_rand_pred
    if os.path.isfile(filename):
        all_results = pd.read_csv(filename,index_col=0)
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[ss_surrogate],'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],
                                  'Beat RP':[beat_rp]}
        new_results = pd.DataFrame(new_results_dictionary)
        
        all_results = all_results.append(new_results, ignore_index = True)
        all_results.to_csv(filename)
    else:
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[ss_surrogate],'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],
                                  'Beat RP':[beat_rp]}
        new_results = pd.DataFrame(new_results_dictionary)
        new_results.to_csv(filename)

def prepare_dataset(patient_folder,data_dimension,fs,low_freq,high_freq,notch_freq,order):
    # Dataset Path
    dataset_path = patient_folder + "all_eeg_dataset.pkl"
    # Datetimes Path
    datetimes_path = patient_folder + "all_datetimes.pkl"
    # Seizure Info Path
    seizure_info_path = patient_folder + "all_seizure_information.pkl"
    # EEG Dataset (Do not contain the 30-minute tolerance time)
    dataset = load_file(dataset_path)
    
    for index,seizure_data in enumerate(dataset):
        dataset[index] = filtering_signals(seizure_data,fs,low_freq,high_freq,notch_freq,order)
        
    # Datetimes
    datetimes = load_file(datetimes_path)
    # Seizure Info
    seizure_onset_datetimes = load_file(seizure_info_path)
    seizure_onset_datetimes = np.array(seizure_onset_datetimes)
    seizure_onset_datetimes = seizure_onset_datetimes[:,0]
    seizure_onset_datetimes = pd.to_datetime(seizure_onset_datetimes,unit='s')
    
    return dataset,datetimes,seizure_onset_datetimes

def get_all_patients_numbers(root_path):
    # Get all patients folders
    all_patients_folders = os.listdir(root_path)
    # Get all patient numbers
    all_patients_numbers = [int(re.findall('\d+', patient_folder)[0]) for patient_folder in all_patients_folders]
    # Sort patients numbers
    all_patients_numbers = np.sort(all_patients_numbers)
    
    return all_patients_numbers

''' Remove seizures that practically do not have any preictal period'''
def remove_datasets_with_small_preictal(dataset,dataset_labels,datetimes,seizure_onset_datetimes):
    
    used_seizure_indexes = []
    
    for seizure_index,seizure_labels in enumerate(dataset_labels):
        preictal_indexes = np.where(seizure_labels==1)
        preictal_indexes = preictal_indexes[0]
        count_preictal_samples = len(preictal_indexes)
        
        if count_preictal_samples>30:
            used_seizure_indexes.append(seizure_index)
            
    new_dataset,new_dataset_labels,new_datetimes,new_seizure_onset_datetimes = [],[],[],[]
    for used_seizure_index in used_seizure_indexes:
        new_dataset.append(dataset[used_seizure_index])
        new_dataset_labels.append(dataset_labels[used_seizure_index])
        new_datetimes.append(datetimes[used_seizure_index])
        new_seizure_onset_datetimes.append(seizure_onset_datetimes[used_seizure_index])
    
    return new_dataset,new_dataset_labels,new_datetimes,new_seizure_onset_datetimes
    
''' Get deep learning model architecture'''
def get_model(nr_filters,filter_size,lstm_units=128):
    swish_function = tf.keras.activations.swish
    input_layer = Input(shape=(2560,19))
    x = Conv1D(nr_filters,filter_size,1,'same')(input_layer)
    x = Conv1D(nr_filters,filter_size,2,'same')(x)
    x = SpatialDropout1D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)

    x = Conv1D(nr_filters*2,filter_size,1,'same')(x)
    x = Conv1D(nr_filters*2,filter_size,2,'same')(x)
    x = SpatialDropout1D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)

    x = Conv1D(nr_filters*4,filter_size,1,'same')(x)
    x = Conv1D(nr_filters*4,filter_size,2,'same')(x)
    x = SpatialDropout1D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(lstm_units,return_sequences=False))(x)
    x = Dropout(0.5)(x)
    x = Dense(2)(x)
    output_layer = Activation('softmax')(x)
                
    
    model = Model(input_layer,output_layer)
    model.compile(optimizer=Adam(3e-4), loss='binary_crossentropy',
                  metrics='acc')
    model.summary()
    
    return model

def remove_sph(y_test,y_pred,seizure_datetimes,sph,seizure_onset_datetime):
    sph_datetime = pd.Timedelta(minutes=10)
    num_windows = len(y_test)
    used_indexes = np.arange(num_windows)
    last_window_datetime = seizure_onset_datetime
    begin_sph_window_index = []
    
    for window_index,window_datetime in enumerate(seizure_datetimes):
        begin_window_datetime = pd.to_datetime(window_datetime[0],unit='s')
        end_window_datetime = pd.to_datetime(window_datetime[-1],unit='s')
        
        if end_window_datetime>(last_window_datetime-sph_datetime):
            begin_sph_window_index.append(window_index)
    
    if len(begin_sph_window_index)>0:
        used_indexes = used_indexes[:begin_sph_window_index[0]]
    
    return y_test[used_indexes],y_pred[used_indexes],seizure_datetimes[used_indexes]

def validate_architecture(dataset,datetimes,seizure_onset_datetimes,training_time,training_ratio,sop,sph,patient_number,batch_size,train_epochs,nr_filters,filter_size,lstm_units=128):
    
    num_seizures = len(dataset)
    print(f'Number of Seizures: {num_seizures}')
    training_seizures = round(training_ratio * num_seizures)
    
    dataset = dataset[0:training_seizures]
    datetimes = datetimes[0:training_seizures]
    seizure_onset_datetimes = seizure_onset_datetimes[0:training_seizures]
    
    sop_gmeans = []
    
    print(f'SOP: {sop} Minutes')
    all_gmeans = []
    # Dataset Labels
    dataset_labels = get_dataset_labels(datetimes, sop, sph)
    # Remove Seizures with Small Preictal
    dataset,dataset_labels,datetimes,seizure_onset_datetimes = remove_datasets_with_small_preictal(dataset,dataset_labels,datetimes,seizure_onset_datetimes)
    
    sub_dataset = dataset[:-1]
    sub_dataset_labels = dataset_labels[:-1]
    sub_datetimes = datetimes[:-1]
    sub_seizure_onset_datetimes = seizure_onset_datetimes[:-1]
    # Get training dataset (only have 4h of data before each training seizure)
    training_data,training_labels,training_datetimes = get_training_dataset(sub_dataset, sub_dataset_labels,
                                                                            sub_datetimes, sub_seizure_onset_datetimes, training_time,
                                                                            1.0, sph,training_seizures)
    # Merge all data
    training_data,training_labels = merge_seizure_datasets(training_data, training_labels)
    # Convert labels into categorical labels (this is necessary to train deep neural networks with softmax)
    training_labels_categorical = to_categorical(training_labels,2)
    # Divide the training data into training and validation sets
    validation_ratio = 0.2
    X_train,X_val,y_train,y_val = train_test_split(training_data,training_labels_categorical,
                                                   test_size=validation_ratio,random_state=random_state,
                                                   stratify=training_labels)

    #------------Train Seizure Prediction Model------------

    print("Train Seizure Prediction Model...")
    # Get standardisation values
    norm_values = [np.mean(X_train),np.std(X_train)]
    # Compute training and validation generators (training generator balances the dataset)
    training_batch_generator = MyCustomGenerator(X_train,y_train,norm_values,batch_size,'training')
    validation_batch_generator = MyCustomGenerator(X_val,y_val,norm_values,batch_size,'validation')
    
    # Construct deep neural network architecture
    model = get_model(nr_filters,filter_size,lstm_units=lstm_units)
        
    train_patience = 50
    
    # Prepare models callbacks (model checkpoint allow the model to be trained until the end selecting the best weights)
    early_stopping_cb = EarlyStopping(monitor='val_loss',patience=train_patience,restore_best_weights=True)
    callbacks_parameters = [early_stopping_cb]
    
    # Get number of training and validation samples
    number_training_samples = len(X_train)
    number_validation_samples = len(X_val)
    # Train the model
    train_history = model.fit(training_batch_generator,steps_per_epoch = len(training_batch_generator),
                                epochs = train_epochs,
                                verbose = 1,
                                validation_data = validation_batch_generator,
                                validation_steps = len(validation_batch_generator),
                                callbacks = callbacks_parameters)
    
    last_epoch = np.argmin(train_history.history['val_loss'])
    #------------Evaluate Model------------
    
    print("Evaluate Seizure Prediction Model...")
    
    # Predict labels
    X_test = (dataset[-1] - norm_values[0]) / norm_values[1]
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    # Get true labels
    y_test = dataset_labels[-1]
    # Get sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    ss = tp/(tp+fn)
    sp = tn/(tn+fp)
    # Save results in arrays
    gmean = np.sqrt(ss*sp)
    
    # Clear variables
    del sub_dataset,sub_dataset_labels,sub_datetimes,sub_seizure_onset_datetimes,training_data,training_datetimes,X_train,X_val,training_batch_generator,validation_batch_generator
    gc.collect()
        
    print("Save Results...")
    # Archive patient results
    filename = f'results_architecture_search_sops_with_strides_{nr_filters}_filters_{filter_size}_{lstm_units}.csv'
    if os.path.isfile(filename):
        all_results = pd.read_csv(filename,index_col=0)
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[ss],
                                  'Specificity':[sp],'G-Mean':[gmean],
                                  'Training Seizures':training_seizures,
                                  'Last Epoch':last_epoch}
        new_results = pd.DataFrame(new_results_dictionary)
        all_results = all_results.append(new_results, ignore_index = True)
        all_results.to_csv(filename)
    else:
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[ss],
                                  'Specificity':[sp],'G-Mean':[gmean],
                                  'Training Seizures':training_seizures,
                                  'Last Epoch':last_epoch}
        new_results = pd.DataFrame(new_results_dictionary)
        new_results.to_csv(filename)
    
    return None
        
#%% Develop and Evaluate Seizure Prediction Model

# Random State
random_state = 42
# Root Path
root_path = "Datasets/"
# root_path = "Not Processed Datasets/"
# Seizure Occurrence (Minutes)
sop_times = [30]*41
# Seizure Prediction Horizon (Minutes)
sph = 10
# Get all patients numbers
all_patient_numbers = get_all_patients_numbers(root_path)
number_patients = len(all_patient_numbers)

for i in range(0,3):
    for patient_index in [0,1,3,6,12,13,15,17,26,40]:
        
        patient_number = all_patient_numbers[patient_index]
        print(f'Patient Number: {patient_number}')
        #------------Get Patient Dataset------------
        
        print("Get Patient Dataset...")
        # Patient Folder
        patient_folder = root_path + "pat_" + str(patient_number) + "/"
        # Prepare Dataset
        data_dimension = '1D'
        dataset,datetimes,seizure_onset_datetimes = prepare_dataset(patient_folder,data_dimension,256,100,0.5,50,4)
        
        
        #------------Select the most optimal SOP------------
        
        print("Selecting the most optimal SOP...")
        # Training Time (SPH does not count)
        training_time = 4
        # Ratio of training and test seizures
        training_ratio = 0.6
        test_ratio = 1 - training_ratio
        
        # Train epochs
        train_epochs = 500
        # Mini-batch size
        batch_size = 8
        # Select the best SOP
        sop = sop_times[patient_index]
        nr_filters = 4 # Number filters of the first layer
        filter_size = 3 # First dimension filter size
        lstm_units = 32
        validate_architecture(dataset,datetimes,seizure_onset_datetimes,training_time,training_ratio,sop,sph,patient_number,batch_size,train_epochs,nr_filters,filter_size,lstm_units)
        # Clear variables
        del dataset,datetimes,seizure_onset_datetimes
        gc.collect()
