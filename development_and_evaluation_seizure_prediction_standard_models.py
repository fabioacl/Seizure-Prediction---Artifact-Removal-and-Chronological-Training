#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:37:49 2021

@author: fabioacl
"""

#%% Import Libraries
import numpy as np
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from seizure_prediction_dataset_batch_generator import SeizurePredictionDatasetGenerator
from compute_eeg_features import ComputeEEGFeatures
import utils
import model_architectures

#%% Develop and Evaluate Seizure Prediction Model

# Random State
random_state = 42
# Root Path
root_path = "Datasets/"
# root_path = "Not Processed Datasets/"

# Firing power threshold
fp_threshold = 0.5
fp_decay_flag = False

# Window duration (Seconds)
window_seconds = 10
number_runs = 31

# Get all patients numbers
all_patient_numbers = utils.get_all_patients_numbers(root_path)
number_patients = len(all_patient_numbers)
model_type = 'Base Model'

if os.path.exists(f'{model_type}/')==False:
    os.mkdir(f'{model_type}/')

for patient_index in range(number_patients):
    
    patient_number = all_patient_numbers[patient_index]
    print(f'Patient Number: {patient_number}')
    
    #------------Get Patient Dataset------------
    
    print("Get Patient Dataset...")
    # Patient Folder
    patient_folder = root_path + "pat_" + str(patient_number) + "/"
    # Prepare Dataset
    fs = 256
    cutoff_freqs = [100,0.5,50]
    filters_orders = [4,4]
    dataset,datetimes,seizure_onset_datetimes = utils.prepare_dataset(patient_folder,fs,cutoff_freqs,filters_orders)
    
    # Dataset Labels
    sop = 30 # Seizure occurrence period
    sph = 10 # Seizure prediction horizon
    dataset_labels = utils.get_dataset_labels(datetimes,seizure_onset_datetimes,sop,sph)
    # Ratio of training and test seizures
    training_ratio = 0.6
    test_ratio = 1 - training_ratio
    num_seizures = len(dataset)
    num_training_seizures = round(num_seizures*training_ratio)
    
    # Remove Seizures with Small Preictal
    print(f'Patient {patient_number} (Before Removing Small Preictal Seizures): {num_seizures}')
    dataset,dataset_labels,datetimes,seizure_onset_datetimes = utils.remove_datasets_with_small_preictal(dataset,dataset_labels,datetimes,seizure_onset_datetimes,sop,sph,window_seconds,fp_threshold,num_training_seizures)
    print(f'Patient {patient_number} (After Removing Small Preictal Seizures): {len(dataset)}')

    print(f'Seizure Occurrence Period: {sop} Minutes')
    # Training Time (SPH does not count)
    training_time = 4
    # Get training dataset (only have 4h of data before each training seizure)
    num_seizures = len(dataset)
    print(f'Number of Seizures: {num_seizures}')
    training_seizures = round(training_ratio * num_seizures)
    
    print(f'Number of Training Seizures: {training_seizures}')
    training_data,training_labels,training_datetimes = utils.get_training_dataset(dataset, dataset_labels,datetimes, seizure_onset_datetimes, training_time, sph, training_seizures)
    # Merge all data
    training_data,training_labels = utils.merge_seizure_datasets(training_data, training_labels)
    # Convert labels into categorical labels (this is necessary to train deep neural networks with softmax)
    training_labels_categorical = to_categorical(training_labels,2)
    
    if 'Features' in model_type:
        feature_groups = ['statistical','spectral band','spectral edge',
                              'hjorth parameters','wavelet','decorrelation time']
        training_data = ComputeEEGFeatures(training_data).calculate_window_features(feature_groups)
            
        num_windows = training_data.shape[0]
        training_data = training_data.reshape((num_windows,-1))
    
    # Create patient directory
    if os.path.exists(f'{model_type}/Patient {patient_number}')==False:
        os.mkdir(f'{model_type}/Patient {patient_number}')
    
    for run_index in range(0,number_runs):
        
        print(f'Run Index: {run_index}')
        # Divide the training data into training and validation sets
        validation_ratio = 0.2
        X_train,X_val,y_train,y_val = train_test_split(training_data,training_labels_categorical,
                                                        test_size=validation_ratio,random_state=run_index,
                                                        stratify=training_labels)
        
        #------------Train Seizure Prediction Model------------
        
        print("Train Seizure Prediction Model...")
        # Get standardisation values
        if 'Features' in model_type:
            norm_values = [np.mean(X_train,axis=0),np.std(X_train,axis=0)]
        else:
            norm_values = [np.mean(X_train),np.std(X_train)]
        np.save(f'{model_type}/Patient {patient_number}/standardisation_values_{patient_number}_{run_index}',norm_values)
        
        # Get mini-batch size
        batch_size = 64
        # Compute training and validation generators (training generator balances the dataset)
        training_batch_generator = SeizurePredictionDatasetGenerator(X_train,y_train,norm_values,batch_size,'training')
        validation_batch_generator = SeizurePredictionDatasetGenerator(X_val,y_val,norm_values,batch_size,'validation')
        
        # Construct artificial neural network architecture
        if 'Features' in model_type:
            nr_features = 'No Reduction'
            dropout_rate = 0.5
            # Multi-GPU
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = model_architectures.get_shallow_model(nr_features,dropout_rate)
        else:
            nr_filters = 128
            filter_size = 3
            lstm_units = 64
            dropout_rate = 0.5
            # Multi-GPU
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = model_architectures.get_deep_model(nr_filters,filter_size,lstm_units,dropout_rate)
        
        train_epochs = 500
        train_patience = round(0.1*train_epochs)
        train_verbose = 1
        
        # Prepare models callbacks (model checkpoint allow the model to be trained until the end selecting the best weights)
        model_checkpoint_cb = ModelCheckpoint(f'{model_type}/Patient {patient_number}/seizure_prediction_model_{patient_number}_{run_index}.h5',
                                              'val_loss', save_best_only=True, verbose=1, mode='min')
        early_stopping_cb = EarlyStopping(monitor='val_loss',patience=train_patience)
        callbacks_parameters = [model_checkpoint_cb,early_stopping_cb]
        
        # Train the model
        train_history = model.fit(training_batch_generator,steps_per_epoch = len(training_batch_generator),
                                  epochs = train_epochs,
                                  verbose = train_verbose,
                                  validation_data = validation_batch_generator,
                                  validation_steps = len(validation_batch_generator),
                                  callbacks = callbacks_parameters)
      
        last_epoch = np.argmin(train_history.history['val_loss'])
        
        #------------Evaluate Model------------
        
        print("Evaluate Seizure Prediction Model...")
        # Load best weights
        model.load_weights(f'{model_type}/Patient {patient_number}/seizure_prediction_model_{patient_number}_{run_index}.h5')

        # Initialise arrays of results
        all_sensitivities = []
        all_fpr_h = []
        all_fp_values = []
        all_alarms = []
        all_pred_labels = []
        all_true_labels = []
        all_datetimes = []
        all_seizure_onset_datetimes = []
        
        for test_index in range(training_seizures,num_seizures):
            # Predict labels
            if 'Features' in model_type:
                X_test = dataset[test_index]
                X_test = ComputeEEGFeatures(X_test).calculate_window_features(feature_groups)
                num_test_windows = X_test.shape[0]
                X_test = X_test.reshape((num_test_windows,-1))
                X_test = (X_test-norm_values[0])/norm_values[1]
            else:
                X_test = (dataset[test_index]-norm_values[0])/norm_values[1]
            
            # If the model did not converge, do not use it.
            if last_epoch>0:
                y_pred = model.predict(X_test)
                y_pred = np.argmax(y_pred,axis=1)
            else:
                nr_samples = X_test.shape[0]
                y_pred = np.zeros((nr_samples,))
                
            # Get true labels
            y_test = dataset_labels[test_index]
            # Get seizure datetimes
            seizure_datetimes = datetimes[test_index]
            # Get seizure onset datetime
            seizure_onset_datetime = seizure_onset_datetimes[test_index]
            # Remove SPH Period
            y_test,y_pred,seizure_datetimes = utils.remove_sph(y_test,y_pred,seizure_datetimes,sph,seizure_onset_datetime)
            # Smooth labels using temporal firing power
            fp_values,filtered_y_pred = utils.temporal_firing_power(y_pred,seizure_datetimes,sop,sph,window_seconds,fp_threshold)
            # Get model evaluation
            true_alarms,false_alarms,possible_firing_time = utils.evaluate_model(filtered_y_pred,y_test,seizure_datetimes,sop,sph,seizure_onset_datetime)
            
            ss = true_alarms
            fpr_h = false_alarms/possible_firing_time
            
            # Save results in lists
            all_sensitivities.append(ss)
            all_fpr_h.append(fpr_h)
            all_fp_values.append(fp_values)
            all_alarms.append(filtered_y_pred)
            all_pred_labels.append(y_pred)
            all_true_labels.append(y_test)
            all_datetimes.append(seizure_datetimes)
            all_seizure_onset_datetimes.append(seizure_onset_datetime)
        
        test_seizures = len(all_sensitivities)
        avg_fpr_h = np.mean(all_fpr_h)
        
        print("Save Results...")
        # Archive patient results
        utils.save_results_array(all_sensitivities,all_fpr_h,all_fp_values,all_alarms,
                                 all_pred_labels,all_true_labels,all_datetimes,last_epoch,
                                 all_seizure_onset_datetimes,patient_number,model_type,run_index)
        
        utils.save_results_csv(patient_number,f'{model_type}/results_{patient_number}_{model_type}.csv',all_sensitivities,
                               all_fpr_h,sop,test_seizures,last_epoch)
        # Clear variables
        del X_train,X_val,training_batch_generator,validation_batch_generator
        gc.collect()
