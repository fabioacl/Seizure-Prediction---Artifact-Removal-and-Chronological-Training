#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:31:59 2022

@author: fabioacl
"""

import os
import numpy as np
import utils

#%% Analyse Data

root_path = "Where model predictions are located (output of training scripts)"

patient_numbers = [402,8902,11002,16202,21902,23902,26102,30802,
                  32702,45402,46702,50802,52302,53402,55202,56402,
                  58602,59102,60002,64702,75202,80702,81102,85202,
                  93402,93902,94402,95202,96002,98102,98202,101702,
                  102202,104602,109502,110602,112802,113902,114702,
                  114902,123902]
number_patients = len(patient_numbers)
sop = 30
sph = 10
fp_decay_flag = False
# Window duration (Seconds)
window_seconds = 10
fp_threshold = 0.5
threshold_type = 'Middle'
model_type = 'Base Model'
chronology_mode = ' Chronological'  # '' or ' Chronological'
already_computed_results = False
number_runs = 31

#%% Application Results (Using Ensemble)

for patient_index in range(number_patients):
    
    patient_number = patient_numbers[patient_index]
    print(f'Patient Number: {patient_number}')
    
    # These patients only contain 3 seizures and, therefore, cannot be used in chronological approaches
    if chronology_mode==' Chronological':
        if patient_number in [52302,81102]:
            continue
    
    all_results = []
    for run_index in range(number_runs):
        print(f'Run: {run_index}')
        
        filepath = f'{root_path}/All Results {model_type}{chronology_mode}/Patient {patient_number}/all_results_{patient_number}_{model_type}_{run_index}.pkl'
        data = utils.load_file(filepath)
        all_pred_labels = data['Predicted Labels']
        all_true_labels = data['True Labels']
        all_datetimes = data['Datetimes']
        all_seizure_onset_datetimes = data['Seizure Onset Datetimes']
        number_test_seizures = len(all_true_labels)
        all_results.append(all_pred_labels)
    
    # Merge the results of all runs for each seizure (voting ensemble)
    all_results = np.array(all_results)
    all_pred_labels = []
    num_tested_seizures = len(all_results[0])
    for i in range(num_tested_seizures):
        y_pred_sz_i = np.round(np.sum(all_results[:,i],axis=0)/31)
        all_pred_labels.append(y_pred_sz_i)
    
    if chronology_mode==' Chronological':
        filepath = f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type} Ensemble/Patient {patient_number}/all_results_{patient_number}_{model_type}.pkl'
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
        y_test = all_true_labels[index]
        seizure_onset_datetime = all_seizure_onset_datetimes[index]
        seizure_datetimes = all_datetimes[index]
        
        # Smooth labels using temporal firing power
        fp_values,filtered_y_pred = utils.temporal_firing_power(y_pred,seizure_datetimes,sop,sph,window_seconds,fp_threshold)
        # Get model evaluation
        true_alarms,false_alarms,possible_firing_time = utils.evaluate_model(filtered_y_pred,y_test,seizure_datetimes,sop,sph,seizure_onset_datetime)
        # Compute surrogate analyses
        surrogate_true_alarms,surrogate_false_alarms,surrogate_possible_firing_time = utils.surrogate_sensitivity(filtered_y_pred, y_test, seizure_datetimes, window_seconds, fp_threshold, sop, sph, seizure_onset_datetime)
        
        all_true_alarms += true_alarms
        all_false_alarms += false_alarms
        all_possible_firing_time += possible_firing_time
        
        all_fp_values.append(fp_values)
        all_alarms.append(filtered_y_pred)
        all_surrogate_true_alarms.append(surrogate_true_alarms)
        all_surrogate_false_alarms.append(surrogate_false_alarms)
        all_surrogate_possible_firing_time.append(surrogate_possible_firing_time)
        
    all_surrogate_true_alarms = np.array(all_surrogate_true_alarms)
    all_surrogate_false_alarms = np.array(all_surrogate_false_alarms)
    all_surrogate_possible_firing_time = np.array(all_surrogate_possible_firing_time)
    all_surrogate_sensitivities = np.sum(all_surrogate_true_alarms,axis=0)/number_test_seizures
    all_surrogate_fpr_h = np.sum(all_surrogate_false_alarms,axis=0)/np.sum(all_surrogate_possible_firing_time,axis=0)
    
    avg_ss = all_true_alarms/number_test_seizures
    avg_fpr_h = all_false_alarms/all_possible_firing_time
    alpha_level = 0.05
    
    # Check if the directory paths exist and if not, create them
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble/')
    
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble/Patient {patient_number}/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/All Results {model_type}{chronology_mode} Ensemble/Patient {patient_number}/')
    
    if os.path.isdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble/')==False:
        os.mkdir(f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble/')
    
    utils.save_ensemble_results_dictionary(avg_ss, avg_fpr_h, all_fp_values, all_alarms, all_surrogate_sensitivities, all_surrogate_fpr_h, all_pred_labels, all_true_labels, all_datetimes, all_seizure_onset_datetimes, root_path, patient_number, model_type, chronology_mode, run_index)
    utils.save_results_ensemble_csv(patient_number,f'{root_path}/Results Seizure Prediction Fixed SOP/{model_type}/Results {model_type}{chronology_mode} Ensemble/results_{model_type}.csv',avg_ss,avg_fpr_h,sop,number_test_seizures,all_surrogate_sensitivities,all_surrogate_fpr_h,alpha_level)
