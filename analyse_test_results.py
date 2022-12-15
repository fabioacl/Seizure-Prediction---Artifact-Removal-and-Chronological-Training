#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:31:59 2022

@author: fabioacl
"""

import os
import numpy as np
import pandas as pd
import pickle
import random
from scipy.special import comb
import scipy.stats

def open_file(filepath):
    with open(filepath,'rb') as object_file:
        data = pickle.load(object_file)
    return data

''' Random Predictor and Surrogate Analysis code were developed by Mauro Pinto'''
def f_RandPredictd(n_SzTotal, s_FPR, d, s_SOP, alpha):
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

'''code that performs surrogate analysis and retrieves its sensitivity
in other words, how many times it predicted the surrogate seizure
in 30 chances'''
def surrogateSensitivity(y_pred,y_true,datetimes,windows_seconds,fp_threshold,sop,sph,seizure_onset_datetime,decay_flag=False):
    
    seizure_true_alarms = []
    seizure_false_alarms = []
    seizure_possible_firing_time = []
    sph_datetime = pd.Timedelta(minutes=sph)
    sop_datetime = pd.Timedelta(minutes=sop)
    #lets do this 30 times
    runs = 30
    for run in range(runs):
        surrogate_labels = np.ones(y_pred.shape)
        surrogate_labels,_ = temporal_firing_power_with_decay(surrogate_labels, datetimes, sop, sph, window_seconds, fp_threshold, decay_flag)
        surrogate_labels,surrogate_seizure_onset_datetime = shuffle_labels(surrogate_labels, datetimes, sop_datetime, sph_datetime, fp_threshold, seizure_onset_datetime)
        new_seizure_true_alarms,new_seizure_false_alarms,new_seizure_possible_firing_time = evaluate_model(y_pred, surrogate_labels, datetimes, sop, sph, surrogate_seizure_onset_datetime)
        seizure_true_alarms.append(new_seizure_true_alarms)
        seizure_false_alarms.append(new_seizure_false_alarms)
        seizure_possible_firing_time.append(new_seizure_possible_firing_time)
    
    return seizure_true_alarms,seizure_false_alarms,seizure_possible_firing_time

def load_file(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

''' The temporal correction consists in introducing 0s in gaps. For each window gap, a 0 is introduced.
    For example, if a gap is longer than 10 seconds, a 0 is introduced in labels array; if a gap is longer than 20 seconds, 
    two 0s are introduced; and so on.'''
def temporal_firing_power_with_decay(y_pred,datetimes,sop,sph,window_seconds,threshold,decay_flag):
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
            step_value = sample_label*fp_step
            if decay_flag:
                coeff_gap = coeff_step - 1 # 1 is the coeff step if everything is ok
                gap_penalty = fp_step*(coeff_gap)
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
    for sample_index in range(0,num_samples):
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

'''Function to evaluate the model.'''
def evaluate_model(y_pred,y_true,datetimes,sop,sph,seizure_onset_datetime):
    sop_time = pd.Timedelta(minutes=sop)
    sph_time = pd.Timedelta(minutes=sph)
    window_datetime_step = pd.Timedelta(nanoseconds=1e9/256)
    first_datetime = pd.to_datetime(datetimes[0][0],unit='s')
    last_sop_datetime = seizure_onset_datetime - sph_time
    begin_sop_datetime = last_sop_datetime - sop_time
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
        
        if current_window_end_datetime >= begin_sop_datetime and current_window_end_datetime < last_sop_datetime:
            inside_sop_time = True
        else:
            inside_sop_time = False
        
        if inside_refractory_time == False and inside_sop_time==False:
            current_window_duration = (current_window_end_datetime - current_window_begin_datetime + window_datetime_step).seconds
            possible_firing_time += current_window_duration
        
        last_window_begin_datetime = current_window_begin_datetime
        last_window_end_datetime = current_window_end_datetime
    
    possible_firing_time /= 3600 # Convert from seconds to hours
    
    for alarm_index in alarm_indexes:
        predicted_label = y_pred[alarm_index]
        true_label = y_true[alarm_index]
        if predicted_label==true_label:
            true_alarms += 1
        elif predicted_label!=true_label:
            false_alarms += 1
    
    return true_alarms,false_alarms,possible_firing_time
        
def save_results_ensemble(patient_number,filename,avg_ss,avg_fpr_h,sop,tested_seizures,ss_rand_pred,all_ss_surrogate,fpr_h_surrogate,last_epoch):
    avg_ss_surrogate = np.mean(all_ss_surrogate)
    std_ss_surrogate = np.std(all_ss_surrogate)
    _,p_value = scipy.stats.ttest_1samp(all_ss_surrogate,avg_ss,alternative='less')
    beat_surrogate = p_value < 0.05 #One sided
    beat_rp = avg_ss>ss_rand_pred
    
    if os.path.isfile(filename):
        all_results = pd.read_csv(filename,index_col=0)
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],'Last Epoch':[last_epoch],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[avg_ss_surrogate],'Sensitivity Std (Surrogate Analysis)':[std_ss_surrogate],
                                  'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],'Beat RP':[beat_rp],
                                  'Beat Surrogate':[beat_surrogate]}
        new_results = pd.DataFrame(new_results_dictionary)
        
        all_results = all_results.append(new_results, ignore_index = True)
        all_results.to_csv(filename)
    else:
        new_results_dictionary = {'Patient':[patient_number],'Sensitivity':[avg_ss],
                                  'FPR/h':[avg_fpr_h],'SOP (Minutes)':[sop],'Last Epoch':[last_epoch],
                                  'Tested Seizures':[tested_seizures],'Sensitivity (Random Prediction)':[ss_rand_pred],
                                  'Sensitivity (Surrogate Analysis)':[avg_ss_surrogate],'Sensitivity Std (Surrogate Analysis)':[std_ss_surrogate],
                                  'FPR/h (Surrogate Analysis)':[fpr_h_surrogate],'Beat RP':[beat_rp],
                                  'Beat Surrogate':[beat_surrogate]}
        new_results = pd.DataFrame(new_results_dictionary)
        new_results.to_csv(filename)

#%% Analyse Data

root_path = "Where model predictions are located (output of training scripts)"

patient_numbers = [402,8902,11002,16202,21902,23902,26102,30802,
                  32702,45402,46702,50802,52302,53402,55202,56402,
                  58602,59102,60002,64702,75202,80702,81102,85202,
                  93402,93902,94402,95202,96002,98102,98202,101702,
                  102202,104602,109502,110602,112802,113902,114702,
                  114902,123902]

sop_times = [30]*41
# Seizure Prediction Horizon (Minutes)
sph = 10
fp_decay_flag = False
# Window duration (Seconds)
window_seconds = 10
fp_threshold = 0.5
threshold_type = 'Middle'
model_type = 'Base Model'
chronology_mode = ' Chronology'  # '' or ' Chronology'
already_computed_results = False
total_seizures = 0
total_hours = 0

#%% Application Results (Using Ensemble)

for patient_index in range(38,41):
    
    patient_number = patient_numbers[patient_index]
    
    if chronology_mode==' Chronology':
        if patient_number in [52302,81102]:
            continue
    
    sop = sop_times[patient_index]
    print(f'Patient Number: {patient_number}')
    
    all_results = []
    for run_index in range(31):
        # print(f'Run: {run_index}')
        filepath = f'{root_path}/All Results {model_type}{chronology_mode}/Patient {patient_number}/all_results_{patient_number}_{model_type}_{run_index}.pkl'
        data = open_file(filepath)
        all_pred_labels = data['Predicted Labels']
        all_true_labels = data['True Labels']
        all_datetimes = data['Datetimes']
        all_seizure_onset_datetimes = data['Seizure Onset Datetimes']
        # all_last_epochs = data['Last Epochs']
        number_test_seizures = len(all_true_labels)
        all_results.append(all_pred_labels)
    
    all_results = np.array(all_results)
    
    all_pred_labels = []
    if len(all_results[0])>0:
        y1 = np.round(np.sum(all_results[:,0],axis=0)/31)
        all_pred_labels.append(y1)
    if len(all_results[0])>1:
        y2 = np.round(np.sum(all_results[:,1],axis=0)/31)
        all_pred_labels.append(y2)
    if len(all_results[0])>2:
        y3 = np.round(np.sum(all_results[:,2],axis=0)/31)
        all_pred_labels.append(y3)
    
    if chronology_mode==' Chronology':
        filepath = f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type} Ensemble Middle Threshold/Patient {patient_number}/all_results_{patient_number}_Base Model.pkl'
        all_pred_labels_normal = np.load(filepath, allow_pickle = True)
        all_true_labels_0 = [all_pred_labels_normal['True Labels'][0]]
        all_true_labels_0.extend(all_true_labels)
        all_true_labels = all_true_labels_0
        
        all_pred_labels_0 = [all_pred_labels_normal['Predicted Labels'][0]]
        all_pred_labels_0.extend(all_pred_labels)
        all_pred_labels = all_pred_labels_0
        
        all_datetimes_0 = [all_pred_labels_normal['Datetimes'][0]]
        all_datetimes_0.extend(all_datetimes)
        all_datetimes = all_datetimes_0
        
        all_seizure_onset_datetimes_0 = [all_pred_labels_normal['Seizure Onset Datetimes'][0]]
        all_seizure_onset_datetimes_0.extend(all_seizure_onset_datetimes)
        all_seizure_onset_datetimes = all_seizure_onset_datetimes_0
        
        number_test_seizures = len(all_true_labels)
    
    all_true_alarms = 0
    all_false_alarms = 0
    all_possible_firing_time = 0
    all_fp_values = []
    all_alarms = []
    all_surrogate_true_alarms = []
    all_surrogate_false_alarms = []
    all_surrogate_possible_firing_time = []
    all_seizure_hours = []

    for index,y_pred in enumerate(all_pred_labels):
        seizure_onset_datetime = all_seizure_onset_datetimes[index]
        seizure_datetimes = all_datetimes[index]
    
        
        total_seizures += 1
        
        # print(f'Seizures: {total_seizures}')
        # print(f'Hours: {total_hours}')
        y_test = all_true_labels[index]
        # Smooth labels using temporal firing power
        fp_values,filtered_y_pred = temporal_firing_power_with_decay(y_pred,seizure_datetimes,sop,sph,window_seconds,fp_threshold,fp_decay_flag)
        # Get model evaluation
        true_alarms,false_alarms,possible_firing_time = evaluate_model(filtered_y_pred,y_test,seizure_datetimes,sop,sph,seizure_onset_datetime)
        print(true_alarms)
        all_true_alarms += true_alarms
        all_false_alarms += false_alarms
        all_possible_firing_time += possible_firing_time
        
        all_fp_values.append(fp_values)
        all_alarms.append(filtered_y_pred)
        
        surrogate_true_alarms,surrogate_false_alarms,surrogate_possible_firing_time = surrogateSensitivity(filtered_y_pred, y_test, seizure_datetimes, window_seconds, fp_threshold, sop, sph, seizure_onset_datetime)
        
        all_surrogate_true_alarms.append(surrogate_true_alarms)
        all_surrogate_false_alarms.append(surrogate_false_alarms)
        all_surrogate_possible_firing_time.append(surrogate_possible_firing_time)
        
    avg_ss = all_true_alarms/number_test_seizures
    avg_fpr_h = all_false_alarms/all_possible_firing_time
    
    print(f'Sensitivity: {avg_ss}')
    print(f'FPR/h: {avg_fpr_h}')
    
    all_surrogate_true_alarms = np.array(all_surrogate_true_alarms)
    all_surrogate_false_alarms = np.array(all_surrogate_false_alarms)
    all_surrogate_possible_firing_time = np.array(all_surrogate_possible_firing_time)
    
    all_surrogate_sensitivities = np.sum(all_surrogate_true_alarms,axis=0)/number_test_seizures
    all_surrogate_fpr_h = np.sum(all_surrogate_false_alarms,axis=0)/np.sum(all_surrogate_possible_firing_time,axis=0)
    
    surrogate_fpr_h = np.mean(all_surrogate_fpr_h)
    
    avg_ss_surrogate = np.mean(all_surrogate_sensitivities)
    std_ss_surrogate = np.std(all_surrogate_sensitivities)
    _,p_value = scipy.stats.ttest_1samp(all_surrogate_sensitivities,avg_ss,alternative='less')
    beat_surrogate = p_value < 0.05 #One sided
    
    print(p_value)
    print(np.mean(all_surrogate_sensitivities))
    print(np.std(all_surrogate_sensitivities))
    print(f'Surrogate Analysis: {beat_surrogate}')
    
    ss_rand_pred = f_RandPredictd(number_test_seizures,avg_fpr_h,1,sop/60,0.05)
    # avg_last_epoch = np.mean(all_last_epochs)
    avg_last_epoch = None
    
    print("Save Results...")
    # Archive patient results
    all_results = {'Sensitivities':avg_ss,
                    'FPR/h':avg_fpr_h,
                    'FP Values':all_fp_values,
                    'All Alarms':all_alarms,
                    'Surrogate Sensitivities':all_surrogate_sensitivities,
                    'Surrogate FPR/h':all_surrogate_fpr_h,
                    # 'Last Epochs':all_last_epochs,
                    'Predicted Labels':all_pred_labels,
                    'True Labels':all_true_labels,
                    'Datetimes':all_datetimes,
                    'Seizure Onset Datetimes':all_seizure_onset_datetimes}
    
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/')
    
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/Patient {patient_number}/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/Patient {patient_number}/')
    
    with open(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/Patient {patient_number}/all_results_{patient_number}_Base Model.pkl','wb') as file:
        pickle.dump(all_results,file)
        
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/')
    
    save_results_ensemble(patient_number,f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble {threshold_type} Threshold/results_Base Model.csv',avg_ss,avg_fpr_h,sop,number_test_seizures,ss_rand_pred,all_surrogate_sensitivities,surrogate_fpr_h,avg_last_epoch)
