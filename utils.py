# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:41:44 2023

@author: fabioacl
"""

import os
import re
import pickle
import random
from scipy.special import comb
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter,filtfilt,iirnotch

def filtering_signals(data,fs,cutoff_freqs,filters_orders):
    """
    Filter signals. It contains a low-pass filter, a high pass filter, and a notch filter
    
    Parameters
    ----------
    data : numpy.array
        Signals.
    fs : int
        Sampling rate.
    cutoff_freqs : list
        Cutoff frequencies for low-pass, high-pass, and notch filters.
    filters_orders : list
        Orders of low-pass and high-pass filters.

    Returns
    -------
    data : numpy.array
        Filtered signals.

    """
    
    low_freq,high_freq,notch_freq = cutoff_freqs
    low_order,high_order = filters_orders
    low_wn = low_freq/(0.5*fs)
    high_wn = high_freq/(0.5*fs)
    b,a = butter(low_order,low_wn,'low')
    data = filtfilt(b,a,data,axis=1)
    b,a = butter(high_order,high_wn,'high')
    data = filtfilt(b,a,data,axis=1)
    b,a = iirnotch(notch_freq,35,fs)
    data = filtfilt(b,a,data,axis=1)
    
    return data

def load_file(filepath):
    """
    Load file

    Parameters
    ----------
    filepath : str
        Filepath.

    Returns
    -------
    data : list
        Dataset.

    """
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def get_dataset_labels(datetimes,seizure_onset_datetimes,sop,sph):
    """
    Get interictal and preictal data. Seizure Prediction Horizon (SPH) represents the interval that
    the patient has to prepare himself/herself to the seizure, i.e., if a SPH of 10 minutes is used,
    the seizure only happens at least 10 minutes after the seizure alarm. Seizure Occurence Period (SOP) 
    represents the interval when the seizure occurs. For predicting the seizure, a preictal interval 
    equals to the SOP is used. Therefore, it is expected that the seizure occurs after the sph and
    inside the following sop interval. For example, if a SOP interval of 40 minutes is used 
    it is expected that the seizure occurs inside the 40 minutes after the SPH.

    Parameters
    ----------
    datetimes : list
        Sample datetimes.
    seizure_onset_datetimes : list
        Seizure onset datetimes.
    sop : int
        Seizure occurrence period.
    sph : int
        Seizure prediction horizon.

    Returns
    -------
    dataset_labels : list
        Sample targets.

    """
    
    num_seizures = len(datetimes)
    sop_datetime = pd.Timedelta(minutes=sop)
    sph_datetime = pd.Timedelta(minutes=sph)
    interictal_indexes = []
    preictal_indexes = []
    sph_indexes = []
    dataset_labels = []
    
    for seizure_index,seizure_onset_datetime in enumerate(seizure_onset_datetimes):
        # Get beginning of pre-ictal period
        begin_sop_seizure_datetime = seizure_onset_datetime - sop_datetime - sph_datetime
        # Get last datetime of SOP
        last_sop_seizure_datetime = begin_sop_seizure_datetime + sop_datetime
        seizure_datetimes = datetimes[seizure_index]
        interictal_indexes.append([])
        preictal_indexes.append([])
        sph_indexes.append([])
        for datetime_index,posix_datetime in enumerate(seizure_datetimes):
            # Get last window datetime
            last_sample_datetime = pd.to_datetime(posix_datetime[-1],unit='s')
            # If the last window datetime is inside SOP, inter-ictal ends and pre-ictal begins
            if begin_sop_seizure_datetime<last_sample_datetime and last_sop_seizure_datetime>=last_sample_datetime:
                preictal_indexes[seizure_index].append(datetime_index)
            elif last_sop_seizure_datetime<last_sample_datetime:
                sph_indexes[seizure_index].append(datetime_index)
            else:
                interictal_indexes[seizure_index].append(datetime_index)

    for seizure_index in range(num_seizures):
        seizure_interictal_begin = interictal_indexes[seizure_index][0]
        seizure_interictal_end = interictal_indexes[seizure_index][-1]
        seizure_length_interictal = seizure_interictal_end - seizure_interictal_begin + 1
        if len(preictal_indexes[seizure_index])>0:
            seizure_preictal_begin = preictal_indexes[seizure_index][0]
            seizure_preictal_end = preictal_indexes[seizure_index][-1]
            seizure_length_preictal = seizure_preictal_end - seizure_preictal_begin + 1
        else:
            seizure_length_preictal = 0
        if len(sph_indexes[seizure_index])>0:
            seizure_sph_begin = sph_indexes[seizure_index][0]
            seizure_sph_end = sph_indexes[seizure_index][-1]
            seizure_length_sph = seizure_sph_end - seizure_sph_begin + 1
        else:
            seizure_length_sph = 0
        # Inter-ictal labels (0)
        seizure_interictal_labels = np.zeros((seizure_length_interictal,))
        # Pre-ictal labels (1)
        seizure_preictal_labels = np.ones((seizure_length_preictal,))
        # SPH labels (0)
        seizure_sph_labels = np.zeros((seizure_length_sph,))
        seizure_dataset_labels = np.concatenate((seizure_interictal_labels,seizure_preictal_labels,seizure_sph_labels))
        
        dataset_labels.append(seizure_dataset_labels)
    
    return dataset_labels

def get_training_dataset(dataset,labels,datetimes,seizure_onset_datetimes,training_time,sph,num_training_seizures):
    """
    Get dataset used for training the models

    Parameters
    ----------
    dataset : list
        Inputs.
    labels : list
        Targets.
    datetimes : list
        Inputs datetimes.
    seizure_onset_datetimes : list
        Seizure onset datetimes.
    training_time : int
        Training time (hours).
    sph : int
        Seizure prediction horizon.
    num_training_seizures : int
        Number of seizures used for training.

    Returns
    -------
    training_dataset : np.array
        Inputs used for training.
    training_labels : np.array
        Targets used for training.
    training_datetimes : np.array
        Datetimes of training samples.

    """
    
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
    """
    Merge samples coming from the different seizures in only one array

    Parameters
    ----------
    dataset : numpy.array
        Inputs per seizure.
    labels : numpy.array
        Targets per seizure.

    Returns
    -------
    merged_dataset : numpy.array
        Inputs.
    merged_labels : numpy.array
        Targets.

    """
    num_seizures = len(dataset)
    merged_dataset = dataset[0]
    merged_labels = labels[0]
    
    for seizure_index in range(1,num_seizures):
        merged_dataset = np.concatenate((merged_dataset,dataset[seizure_index]))
        merged_labels = np.concatenate((merged_labels,labels[seizure_index]))
    
    return merged_dataset,merged_labels

def temporal_firing_power(y_pred,datetimes,sop,sph,window_seconds,threshold):
    """
    The temporal correction consists of considering gaps as 0s. In other words, 
    everytime there is a gap nothing is added to the firing power filter but the 
    time continues counting.

    Parameters
    ----------
    y_pred : numpy.array
        Predicted targets.
    datetimes : numpy.array
        Sample datetimes.
    sop : int
        Seizure occurrence period.
    sph : int
        Seizure prediction horizon.
    window_seconds : int
        Number of seconds per window (sample).
    threshold : float
        Firing power regulariser threshold.

    Returns
    -------
    firing_power_windows : numpy.array
        Firing power values.
    filtered_y_pred : numpy.array
        Post-processed target values.

    """
    
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
            step_value = sample_label*fp_step
                
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

def prepare_dataset(patient_folder,fs,cutoff_freqs,filters_orders):
    """
    Load data and preprocess them

    Parameters
    ----------
    patient_folder : str
        Patient folderpath.
    fs : int
        Sampling rate.
    cutoff_freqs : list
        Cutoff frequencies for low-pass, high-pass, and notch filters.
    filters_orders : list
        Orders of low-pass and high-pass filters.

    Returns
    -------
    dataset : list
        Inputs.
    datetimes : list
        Samples datetimes.
    seizure_onset_datetimes : list
        Seizure onset datetimes.

    """
    # Dataset Path
    dataset_path = patient_folder + "all_eeg_dataset.pkl"
    # Datetimes Path
    datetimes_path = patient_folder + "all_datetimes.pkl"
    # Seizure Info Path
    seizure_info_path = patient_folder + "all_seizure_information.pkl"
    # EEG Dataset (Do not contain the 30-minute tolerance time)
    dataset = load_file(dataset_path)
    
    for index,seizure_data in enumerate(dataset):
        dataset[index] = filtering_signals(seizure_data,fs,cutoff_freqs,filters_orders)
        
    # Datetimes
    datetimes = load_file(datetimes_path)
    # Seizure onset datetimes
    seizure_onset_datetimes = load_file(seizure_info_path)
    seizure_onset_datetimes = np.array(seizure_onset_datetimes)
    seizure_onset_datetimes = seizure_onset_datetimes[:,0]
    seizure_onset_datetimes = pd.to_datetime(seizure_onset_datetimes,unit='s')
    
    return dataset,datetimes,seizure_onset_datetimes

def get_all_patients_numbers(root_path):
    """
    Get all patients ids available in the main dataset

    Parameters
    ----------
    root_path : str
        Directory path of the folder that contains patients datasets.

    Returns
    -------
    all_patients_numbers : list
        All available patients ids.

    """
    # Get all patients folders
    all_patients_folders = os.listdir(root_path)
    # Get all patient numbers
    all_patients_numbers = [int(re.findall('\d+', patient_folder)[0]) for patient_folder in all_patients_folders]
    # Sort patients numbers
    all_patients_numbers = np.sort(all_patients_numbers)
    
    return all_patients_numbers

''' Remove seizures that practically do not have any preictal period'''
def remove_datasets_with_small_preictal(dataset,dataset_labels,datetimes,seizure_onset_datetimes,sop,sph,window_seconds,fp_threshold,num_training_seizures):
    """
    Remove seizures that practically do not have any preictal period

    Parameters
    ----------
    dataset : list
        Inputs.
    dataset_labels : list
        Targets.
    datetimes : list
        Sample datetimes.
    seizure_onset_datetimes : list
        Seizure onset datetimes.
    sop : int
        Seizure onset period.
    sph : int
        Seizure prediction horizon.
    window_seconds : int
        Number of seconds per window (sample).
    fp_threshold : float
        Firing power regulariser threshold.
    num_training_seizures : int
        Number of training seizures.

    Returns
    -------
    new_dataset : list
        Pre-processed inputs.
    new_dataset_labels : list
        Pre-processed targets.
    new_datetimes : list
        Pre-processed samples datetimes.
    new_seizure_onset_datetimes : list
        Pre-processed seizure onset datetimes.

    """
    
    used_seizure_indexes = []
    
    for seizure_index,seizure_labels in enumerate(dataset_labels):
        
        # Smooth labels using temporal firing power
        seizure_datetimes = datetimes[seizure_index]
        seizure_onset_datetime = seizure_onset_datetimes[seizure_index]
        fp_values,filtered_y_pred = temporal_firing_power(seizure_labels,seizure_datetimes,sop,sph,window_seconds,fp_threshold)
        # Get model evaluation
        ss,fpr_h,_ = evaluate_model(filtered_y_pred,seizure_labels,seizure_datetimes,sop,sph,seizure_onset_datetime)
        if ss==1 or num_training_seizures>seizure_index:
            used_seizure_indexes.append(seizure_index)
            
    new_dataset,new_dataset_labels,new_datetimes,new_seizure_onset_datetimes = [],[],[],[]
    for used_seizure_index in used_seizure_indexes:
        new_dataset.append(dataset[used_seizure_index])
        new_dataset_labels.append(dataset_labels[used_seizure_index])
        new_datetimes.append(datetimes[used_seizure_index])
        new_seizure_onset_datetimes.append(seizure_onset_datetimes[used_seizure_index])
        
    return new_dataset,new_dataset_labels,new_datetimes,new_seizure_onset_datetimes

def evaluate_model(y_pred,y_true,datetimes,sop,sph,seizure_onset_datetime):
    """
    Evaluate the model. The data do not contain the SPH.

    Parameters
    ----------
    y_pred : numpy.array
        Predicted targets.
    y_true : numpy.array
        True targets.
    datetimes : numpy.array
        Sample datetimes.
    sop : int
        Seizure occurrence period.
    sph : int
        Seizure prediction horizon.
    seizure_onset_datetime : pandas.Timestamp
        Seizure onset datetime.

    Returns
    -------
    sensitivity : float
        Seizure sensitivity.
    fpr_h : float
        False prediction rate per hour.
    possible_firing_time : float
        Number of hours in which the model is able to fire an alarm.

    """
    
    sop_time = pd.Timedelta(minutes=sop)
    sph_time = pd.Timedelta(minutes=sph)
    window_datetime_step = pd.Timedelta(nanoseconds=1e9/256)
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
    # FPR/h is computed over the time where is possible to fire alarms
    fpr_h = false_alarms/possible_firing_time
    
    return sensitivity,fpr_h,possible_firing_time

''' Random Predictor and Surrogate Analysis code were developed by Mauro Pinto'''
def random_predictor(n_SzTotal, s_FPR, d, s_SOP, alpha):
    """
    Random Predictor developed by Mauro Pinto (one of the coauthors)

    Parameters
    ----------
    n_SzTotal : int
        Number of tested seizures.
    s_FPR : float
        Obtained FPR/h.
    d : int
        Number of models.
    s_SOP : int
        Seizure occurrence period.
    alpha : float
        Significance level.

    Returns
    -------
    s_kmax : float
        Seizure sensitivity of random predictor.

    """
    # Random predictor with d free independent parameters

    v_PBinom = np.zeros(n_SzTotal)
    s_kmax = 0
    
    
    # o +1, -1 tem a ver com no matlab a iteracao comeca em 1, aqui em 0 :)
    for seizure_i in range(0,n_SzTotal):
        v_Binom=comb(n_SzTotal,seizure_i+1)
        s_PPoi=s_FPR*s_SOP
        v_PBinom[seizure_i]=v_Binom*s_PPoi**(seizure_i+1)*((1-s_PPoi)**(n_SzTotal-seizure_i-1))
        
    v_SumSignif=1-(1-np.cumsum(np.flip(v_PBinom)))**d>alpha
    s_kmax=np.count_nonzero(v_SumSignif)/n_SzTotal
    
    return s_kmax

# code to shuffle the pre-seizure labels for the surrogate
def shuffle_labels(surrogate_labels,datetimes,sop_datetime,sph_datetime,fp_threshold,seizure_onset_datetime):
    """
    Code to shuffle the pre-seizure labels for the surrogate developed by Mauro Pinto (one of the coauthors)

    Parameters
    ----------
    surrogate_labels : numpy.array
        Targets used in surrogate analysis.
    datetimes : numpy.array
        Sample datetimes.
    sop_datetime : int
        Seizure occurrence period in minutes.
    sph_datetime : int
        Seizure prediction horizon in minutes.
    fp_threshold : float
        Firing power threshold.
    seizure_onset_datetime : pandas.Timestamp
        Seizure onset datetime.

    Returns
    -------
    surrogate_labels : numpy.array
        Pre-processed targets used in surrogate analysis.
    surrogate_seizure_onset_datetime : TYPE
        First datetime from which the model is able to fire alarms.

    """
    
    # Surrogate analysis could not start inside the truth preictal
    end_alarms_datetime = seizure_onset_datetime - sop_datetime - sph_datetime
    possible_surrogate_indexes = []
    
    # This process is performed because we can only use the surrogate in the periods where the alarms
    # can be fired. The surrogate can not be fired at the beginning and also cannot be fired after a 
    # long gap because of the temporal decay.
    for window_index,fp_value in enumerate(surrogate_labels):
        
        current_window_end_datetime = pd.to_datetime(datetimes[window_index][-1],unit='s')
        
        # If it is possible to fire alarm, we can also make the surrogate analysis and has to finish before the true SOP.
        if fp_value >= fp_threshold and current_window_end_datetime < (end_alarms_datetime - sop_datetime):
            possible_surrogate_indexes.append(window_index)
    
    # Lets just take one random index and build the surrogate prediction.
    sop_begin_index = random.sample(possible_surrogate_indexes,1)[0]
    sop_begin_datetime = pd.to_datetime(datetimes[sop_begin_index][0],unit='s')
    sop_end_datetime = sop_begin_datetime + sop_datetime
    
    surrogate_preictal_indexes = []
    last_window_begin_datetime = pd.to_datetime(datetimes[sop_begin_index][0],unit='s')
    num_windows = len(surrogate_labels)
    # The second condition is used when there are some missing windows at the end of the list and the
    # last_window_begin_datetime is still before the sop_end_datetime.
    while last_window_begin_datetime < sop_end_datetime and sop_begin_index<num_windows:
        surrogate_preictal_indexes.append(sop_begin_index)
        last_window_begin_datetime = pd.to_datetime(datetimes[sop_begin_index][0],unit='s')
        sop_begin_index += 1
    
    surrogate_labels[:] = 0
    surrogate_preictal_indexes = np.array(surrogate_preictal_indexes)
    surrogate_labels[surrogate_preictal_indexes] = 1
    surrogate_seizure_onset_datetime = sop_end_datetime + sph_datetime
        
    return surrogate_labels,surrogate_seizure_onset_datetime

def surrogate_sensitivity(y_pred,y_true,datetimes,window_seconds,fp_threshold,sop,sph,seizure_onset_datetime,decay_flag=False):
    """
    Code that performs surrogate analysis and retrieves its sensitivity in other words, 
    how many times it predicted the surrogate seizure in 30 chances. Developed by Mauro Pinto (one of the coauthors)

    Parameters
    ----------
    y_pred : numpy.array
        Predicted targets.
    y_true : numpy.array
        True targets.
    datetimes : numpy.array
        Sample datetimes.
    window_seconds : int
        Length of windows in seconds.
    fp_threshold : float
        Firing power threshold.
    sop : int
        Seizure occurrence period in minutes.
    sph : int
        Seizure prediction horizon in minutes.
    seizure_onset_datetime : pandas.Timestamp
        Seizure onset datetime.

    Returns
    -------
    seizure_true_alarms : list
        Number of true alarms.
    seizure_false_alarms : list
        Number of false alarms.
    seizure_possible_firing_time : list
        Number of hours in which the model is able to fire an alarm.

    """
    
    seizure_true_alarms = []
    seizure_false_alarms = []
    seizure_possible_firing_time = []
    sph_datetime = pd.Timedelta(minutes=sph)
    sop_datetime = pd.Timedelta(minutes=sop)
    #lets do this 30 times
    runs = 30
    for run in range(runs):
        surrogate_labels = np.ones(y_pred.shape)
        surrogate_labels,_ = temporal_firing_power(surrogate_labels, datetimes, sop, sph, window_seconds, fp_threshold)
        surrogate_labels,surrogate_seizure_onset_datetime = shuffle_labels(surrogate_labels, datetimes, sop_datetime, sph_datetime, fp_threshold, seizure_onset_datetime)
        new_seizure_true_alarms,new_seizure_false_alarms,new_seizure_possible_firing_time = evaluate_model(y_pred, surrogate_labels, datetimes, sop, sph, surrogate_seizure_onset_datetime)
        seizure_true_alarms.append(new_seizure_true_alarms)
        seizure_false_alarms.append(new_seizure_false_alarms)
        seizure_possible_firing_time.append(new_seizure_possible_firing_time)
    
    return seizure_true_alarms,seizure_false_alarms,seizure_possible_firing_time

def remove_sph(y_test,y_pred,seizure_datetimes,sph,seizure_onset_datetime):
    """
    Remove seizure prediction horizon of seizure used for testing since it should not be used to evaluate the model

    Parameters
    ----------
    y_test : numpy.array
        True targets.
    y_pred : numpy.array
        Predicted targets.
    seizure_datetimes : numpy.array
        Samples datetimes in the considered seizure.
    sph : int
        Seizure prediction horizon.
    seizure_onset_datetime : pandas.Timestamp
        Seizure onset datetime.

    Returns
    -------
    y_test : numpy.array
        True targets without SPH.
    y_pred : numpy.array
        Predicted targets without SPH.
    seizure_datetimes : numpy.array
        Samples datetimes in the considered seizure without SPH.

    """
    
    sph_datetime = pd.Timedelta(minutes=10)
    num_windows = len(y_test)
    used_indexes = np.arange(num_windows)
    last_window_datetime = seizure_onset_datetime
    begin_sph_window_index = []
    
    for window_index,window_datetime in enumerate(seizure_datetimes):
        end_window_datetime = pd.to_datetime(window_datetime[-1],unit='s')
        
        if end_window_datetime>(last_window_datetime-sph_datetime):
            begin_sph_window_index.append(window_index)
    
    if len(begin_sph_window_index)>0:
        used_indexes = used_indexes[:begin_sph_window_index[0]]
        
    y_test = y_test[used_indexes]
    y_pred = y_pred[used_indexes]
    seizure_datetimes = seizure_datetimes[used_indexes]
    
    return y_test,y_pred,seizure_datetimes

def save_results_csv(patient_number,filename,all_sensitivities,all_fpr_h,sop,tested_seizures,last_epoch):
    """
    Save results in CSV file

    Parameters
    ----------
    patient_number : int
        Patient ID.
    filename : str
        CSV Filename.
    all_sensitivities : list
        All obtained seizure sensitivities.
    all_fpr_h : list
        All obtained FPR/h's.
    sop : int
        Seizure occurrence period.
    tested_seizures : int
        Number of tested seizures.
    last_epoch : int or float
        Number of epochs used for training the model.

    Returns
    -------
    None.

    """
    
    avg_ss = np.mean(all_sensitivities)
    avg_fpr_h = np.mean(all_fpr_h)
    if os.path.isfile(filename):
        all_results = pd.read_csv(filename,index_col=0)
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],'Last Epoch':[last_epoch],
                                  'Tested Seizures':[tested_seizures]}
        new_results = pd.DataFrame(new_results_dictionary)
        
        all_results = all_results.append(new_results, ignore_index = True)
        all_results.to_csv(filename)
    else:
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],'Last Epoch':[last_epoch],
                                  'Tested Seizures':[tested_seizures]}
        new_results = pd.DataFrame(new_results_dictionary)
        new_results.to_csv(filename)
    
    return None

def save_results_ensemble_csv(patient_number,filename,avg_ss,avg_fpr_h,sop,tested_seizures,all_surrogate_sensitivities,all_surrogate_fpr_h,alpha_level):
    """
    Save results obtained after merging outputs from all models in a CSV file

    Parameters
    ----------
    patient_number : int
        Patient ID.
    filename : str
        CSV Filename.
    avg_ss : float
        Average seizure sensitivity.
    avg_fpr_h : float
        Average FPR/h.
    sop : int
        Seizure occurrence period.
    tested_seizures : int
        Number of tested seizures.
    all_surrogate_sensitivities : list
        Computed surrogate sensitivities.
    all_surrogate_fpr_h : list
        Computed surrogate FPR/h.
    alpha_level : float
        Significance level.

    Returns
    -------
    None.

    """
    
    avg_ss_surrogate = np.mean(all_surrogate_sensitivities)
    std_ss_surrogate = np.std(all_surrogate_sensitivities)
    fpr_h_surrogate = np.mean(all_surrogate_fpr_h)
    _,p_value = scipy.stats.ttest_1samp(all_surrogate_sensitivities,avg_ss,alternative='less')
    beat_surrogate = p_value < alpha_level # One sided
    ss_rand_pred = random_predictor(tested_seizures,avg_fpr_h,1,sop/60,alpha_level)
    
    avg_ss_surrogate = np.mean(all_surrogate_sensitivities)
    std_ss_surrogate = np.std(all_surrogate_sensitivities)
    _,p_value = scipy.stats.ttest_1samp(all_surrogate_sensitivities,avg_ss,alternative='less')
    beat_surrogate = p_value < 0.05 #One sided
    beat_rp = avg_ss>ss_rand_pred
    
    if os.path.isfile(filename):
        all_results = pd.read_csv(filename,index_col=0)
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[avg_ss_surrogate],'Sensitivity Std (Surrogate Analysis)':[std_ss_surrogate],
                                  'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],'Beat RP':[beat_rp],
                                  'Beat Surrogate':[beat_surrogate]}
        new_results = pd.DataFrame(new_results_dictionary)
        
        all_results = all_results.append(new_results, ignore_index = True)
        all_results.to_csv(filename)
    else:
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[avg_ss_surrogate],'Sensitivity Std (Surrogate Analysis)':[std_ss_surrogate],
                                  'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],'Beat RP':[beat_rp],
                                  'Beat Surrogate':[beat_surrogate]}
        new_results = pd.DataFrame(new_results_dictionary)
        new_results.to_csv(filename)
    
    return None

def save_results_dictionary(all_sensitivities,all_fpr_h,all_fp_values,all_alarms,all_pred_labels,all_true_labels,all_datetimes,last_epoch,all_seizure_onset_datetimes,patient_number,model_type,run_index):
    """
    Save results in a dictionary object

    Parameters
    ----------
    all_sensitivities : list
        Seizure sensitivities.
    all_fpr_h : list
        FPR/h.
    all_fp_values : list
        Firing power values.
    all_alarms : list
        Alarms.
    all_pred_labels : list
        Predicted targets.
    all_true_labels : list
        True targets.
    all_datetimes : list
        Sample datetimes.
    last_epoch : int
        Number of epochs used for training the model.
    all_seizure_onset_datetimes : list
        Seizures onsets datetimes.
    patient_number : int
        Patient ID.
    model_type : str
        Type of architecture used to develop the seizure prediction model.
    run_index : int
        Current run.

    Returns
    -------
    None.

    """
    all_results = {'Sensitivities':all_sensitivities,
                   'FPR/h':all_fpr_h,
                   'FP Values':all_fp_values,
                   'All Alarms':all_alarms,
                   'Predicted Labels':all_pred_labels,
                   'True Labels':all_true_labels,
                   'Datetimes':all_datetimes,
                   'Last Epoch':last_epoch,
                   'Seizure Onset Datetimes':all_seizure_onset_datetimes}
        
    with open(f'{model_type}/Patient {patient_number}/all_results_{patient_number}_{model_type}_{run_index}.pkl','wb') as file:
        pickle.dump(all_results,file)
    
    return None

def save_ensemble_results_dictionary(avg_ss,avg_fpr_h,all_fp_values,all_alarms,all_surrogate_sensitivities,
                                     all_surrogate_fpr_h,all_pred_labels,all_true_labels,all_datetimes,
                                     all_seizure_onset_datetimes,root_path,patient_number,model_type,chronology_mode,run_index):
    """
    Save results obtained after merging outputs from all models in a dictionary object
    
    Parameters
    ----------
    avg_ss : float
        Average seizure sensitivity.
    all_fpr_h : float
        Average FPR/h.
    all_fp_values : list
        Firing power values.
    all_alarms : list
        Alarms.
    all_surrogate_sensitivities : list
        Seizure sensitivities obtained in surrogate analysis.
    all_surrogate_fpr_h : list
        FPR/h obtained in surrogate analysis.
    all_pred_labels : list
        Predicted targets.
    all_true_labels : list
        True targets.
    all_datetimes : list
        Sample datetimes.
    all_seizure_onset_datetimes : list
        Seizures onsets datetimes.
    root_path : str
        Main directory to save the results.
    patient_number : int
        Patient ID.
    model_type : str
        Type of architecture used to develop the seizure prediction model.
    chronology_mode : str
        Type of model regarding training chronology
    run_index : int
        Current run.

    Returns
    -------
    None.

    """
    all_results = {'Sensitivities':avg_ss,
                    'FPR/h':avg_fpr_h,
                    'FP Values':all_fp_values,
                    'All Alarms':all_alarms,
                    'Surrogate Sensitivities':all_surrogate_sensitivities,
                    'Surrogate FPR/h':all_surrogate_fpr_h,
                    'Predicted Labels':all_pred_labels,
                    'True Labels':all_true_labels,
                    'Datetimes':all_datetimes,
                    'Seizure Onset Datetimes':all_seizure_onset_datetimes}
    
    with open(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble/Patient {patient_number}/all_results_{patient_number}_{model_type}.pkl','wb') as file:
        pickle.dump(all_results,file)
    
    return None





