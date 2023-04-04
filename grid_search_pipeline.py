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
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,SpatialDropout1D,Activation,BatchNormalization,Bidirectional,LSTM,Dropout,Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from seizure_prediction_dataset_batch_generator import SeizurePredictionDatasetGenerator
from compute_eeg_features import ComputeEEGFeatures
import utils

#%% Functions
    
''' Get deep learning model architecture'''
def get_deep_model(nr_filters,filter_size,lstm_units=128):
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

def get_shallow_model(nr_features):
    
    swish_function = tf.keras.activations.swish
    features_input_layer = Input(shape=(1045,))
    handcrafted_features = Dropout(0.5)(features_input_layer)
    
    if nr_features!='No Reduction':
        handcrafted_features = Dense(nr_features)(handcrafted_features)
        handcrafted_features = Activation(swish_function)(handcrafted_features)
    
    x = Dropout(0.5)(handcrafted_features)
    
    x = Dense(2)(x)
    output_layer = Activation('softmax')(x)
        
    model = Model(features_input_layer,output_layer)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy',metrics='acc')
    model.summary()
    
    return model
        
#%% Develop and Evaluate Seizure Prediction Model

# Random State
random_state = 42
# Root Path
root_path = "Datasets/"
# root_path = "Not Processed Datasets/"
architecture_type = 'shallow'
number_runs = 3

# Get all patients numbers
all_patient_numbers = utils.get_all_patients_numbers(root_path)
number_patients = len(all_patient_numbers)

for i in range(0,number_runs):
    for patient_index in [0,1,3,6,12,13,15,17,26,40]:
        
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
        
        # SOP, SPH, and training time (SPH does not count)
        sop = 30
        sph = 10
        training_time = 4
        # Ratio of training and test seizures
        training_ratio = 0.6
        test_ratio = 1 - training_ratio
        num_seizures = len(dataset)
        print(f'Number of Seizures: {num_seizures}')
        training_seizures = round(training_ratio * num_seizures)
        
        # Remove the seizures that are going to be used for testing in future
        dataset = dataset[0:training_seizures]
        datetimes = datetimes[0:training_seizures]
        seizure_onset_datetimes = seizure_onset_datetimes[0:training_seizures]
        
        sop_gmeans = []
        
        print(f'SOP: {sop} Minutes')
        all_gmeans = []
        # Dataset Labels
        dataset_labels = utils.get_dataset_labels(datetimes,seizure_onset_datetimes,sop,sph)
        # Remove Seizures with Small Preictal
        fp_threshold = 0.5
        window_seconds = 10
        dataset,dataset_labels,datetimes,seizure_onset_datetimes = utils.remove_datasets_with_small_preictal(dataset,dataset_labels,datetimes,
                                                                                                             seizure_onset_datetimes,sop,sph,window_seconds,
                                                                                                             fp_threshold,training_seizures)
        # Training dataset (last seizure will be used for evaluation)
        sub_dataset = dataset[:-1]
        sub_dataset_labels = dataset_labels[:-1]
        sub_datetimes = datetimes[:-1]
        sub_seizure_onset_datetimes = seizure_onset_datetimes[:-1]
        # Get training dataset (only have 4h of data before each training seizure)
        training_data,training_labels,training_datetimes = utils.get_training_dataset(sub_dataset, sub_dataset_labels,
                                                                                      sub_datetimes, sub_seizure_onset_datetimes, 
                                                                                      training_time,sph,training_seizures)
        # Merge all data
        training_data,training_labels = utils.merge_seizure_datasets(training_data, training_labels)
        # Convert labels into categorical labels (this is necessary to train deep neural networks with softmax)
        training_labels_categorical = to_categorical(training_labels,2)
        
        # If shallow neural network, compute features
        if architecture_type=='shallow':
            feature_groups = ['statistical','spectral band','spectral edge',
                                  'hjorth parameters','wavelet','decorrelation time']
            training_data = ComputeEEGFeatures(training_data).calculate_window_features(feature_groups)
            num_windows = training_data.shape[0]
            training_data = training_data.reshape((num_windows,-1))
        
        # Divide the training data into training and validation sets
        validation_ratio = 0.2
        X_train,X_val,y_train,y_val = train_test_split(training_data,training_labels_categorical,
                                                       test_size=validation_ratio,random_state=random_state,
                                                       stratify=training_labels)
    
        #------------Train Seizure Prediction Model------------
    
        print("Train Seizure Prediction Model...")
        # Get standardisation values
        if architecture_type=='deep':
            norm_values = [np.mean(X_train),np.std(X_train)]
        else:
            norm_values = [np.mean(X_train,axis=0),np.std(X_train,axis=0)]
            
        # Compute training and validation generators (training generator balances the dataset)
        batch_size = 8
        training_batch_generator = SeizurePredictionDatasetGenerator(X_train,y_train,norm_values,batch_size,'training')
        validation_batch_generator = SeizurePredictionDatasetGenerator(X_val,y_val,norm_values,batch_size,'validation')
        
        # Construct artificial neural network architecture
        if architecture_type=='deep':
            nr_filters = 4 # Number filters of the first layer
            filter_size = 3 # First dimension filter size
            lstm_units = 32 # Number of LSTM units
            model = get_deep_model(nr_filters,filter_size,lstm_units=lstm_units)
        else:
            nr_features = 'No Reduction'
            model = get_shallow_model(nr_features)
        
            
        train_epochs = 500
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
        if architecture_type=='deep':
            X_eval = (dataset[-1] - norm_values[0]) / norm_values[1]
        else:
            X_eval = dataset[-1]
            X_eval = ComputeEEGFeatures(X_eval).calculate_window_features(feature_groups)
            num_test_windows = X_eval.shape[0]
            X_eval = X_eval.reshape((num_test_windows,-1))
            X_eval = (X_eval-norm_values[0])/norm_values[1]
            
        y_pred = model.predict(X_eval)
        y_pred = np.argmax(y_pred,axis=1)
        # Get true labels
        y_eval = dataset_labels[-1]
        # Get sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_eval,y_pred).ravel()
        ss = tp/(tp+fn)
        sp = tn/(tn+fp)
        # Save results in arrays
        gmean = np.sqrt(ss*sp)
        
        # Clear variables
        del sub_dataset,sub_dataset_labels,sub_datetimes,sub_seizure_onset_datetimes,training_data,training_datetimes,X_train,X_val,training_batch_generator,validation_batch_generator
        gc.collect()
            
        print("Save Results...")
        # Archive patient results
        if architecture_type=='deep':
            filename = f'results_architecture_search_sops_with_strides_{nr_filters}_filters_{filter_size}_{lstm_units}.csv'
        else:
            filename = f'results_architecture_search_sops_with_strides_{nr_features}.csv'
            
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
            
        # Clear variables
        del dataset,datetimes,seizure_onset_datetimes
        gc.collect()
