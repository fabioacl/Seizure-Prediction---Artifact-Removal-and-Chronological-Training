# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:11:19 2023

@author: costaf
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,SpatialDropout1D,Activation,BatchNormalization,Bidirectional,LSTM,Dropout,Dense
from tensorflow.keras.optimizers import Adam

def get_deep_model(nr_filters,filter_size,lstm_units,dropout_rate):
    """
    Build deep neural network architecture

    Parameters
    ----------
    nr_filters : int
        Number of filters used in the first convolutional layers.
    filter_size : int
        Number of weights per convolutional filter.
    lstm_units : int
        LSTM units.
    dropout_rate : float
        Dropout rate.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Deep neural network architecture.
    """

    swish_function = tf.keras.activations.swish
    input_layer = Input(shape=(2560,19))
    
    x = Conv1D(nr_filters,filter_size,1,'same')(input_layer)
    x = Conv1D(nr_filters,filter_size,2,'same')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(nr_filters*2,filter_size,1,'same')(x)
    x = Conv1D(nr_filters*2,filter_size,2,'same')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(nr_filters*4,filter_size,1,'same')(x)
    x = Conv1D(nr_filters*4,filter_size,2,'same')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(lstm_units,return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(2)(x)
    output_layer = Activation('softmax')(x)

    
    model = Model(input_layer,output_layer)
    model.compile(optimizer=Adam(3e-4), loss='binary_crossentropy',metrics='acc')
    model.summary()
    
    return model

def get_shallow_model(nr_features,dropout_rate):
    """
    Build deep neural network architecture

    Parameters
    ----------
    nr_features : int
        Number of neurons in the hidden layer
    dropout_rate : float
        Dropout rate.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Shallow neural network architecture.
    """
    
    swish_function = tf.keras.activations.swish
    features_input_layer = Input(shape=(1083,))
    handcrafted_features = Dropout(dropout_rate)(features_input_layer)
    
    if nr_features!='No Reduction':
        handcrafted_features = Dense(nr_features)(handcrafted_features)
        handcrafted_features = Activation(swish_function)(handcrafted_features)
        x = Dropout(dropout_rate)(handcrafted_features)
    
    x = Dense(2)(x)
    output_layer = Activation('softmax')(x)
        
    model = Model(features_input_layer,output_layer)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy',metrics='acc')
    model.summary()
    
    return model
