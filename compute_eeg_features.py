#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:23:07 2021

@author: fabioacl
"""

import numpy as np
import scipy
import pywt
import itertools

class ComputeEEGFeatures():
    
    def __init__(self,dataset):
        self.dataset = dataset
    
    def calculate_window_features(self,feature_groups):
        
        dataset_features = []
        for window in self.dataset:
            nr_channels = window.shape[1]
            window_features = []
            for window_channel_index in range(nr_channels):
                window_channel = np.squeeze(window[:,window_channel_index])
                window_channel_features = []
                if 'statistical' in feature_groups:
                    statistical_features = self.compute_statistical_features(window_channel)
                    window_channel_features.extend(statistical_features)
                if 'spectral band' in feature_groups:
                    spectral_bands_features = self.compute_spectral_bands_features(window_channel)
                    window_channel_features.extend(spectral_bands_features)
                if 'spectral edge' in feature_groups:
                    spectral_edge_features = self.compute_spectral_edge_frequency_features(window_channel)
                    window_channel_features.extend(spectral_edge_features)
                if 'hjorth parameters' in feature_groups:
                    hjorth_parameters_features = self.compute_hjorth_parameters_features(window_channel)
                    window_channel_features.extend(hjorth_parameters_features)
                if 'wavelet' in feature_groups:
                    wavelet_features = self.compute_wavelet_features(window_channel)
                    window_channel_features.extend(wavelet_features)
                if 'decorrelation time' in feature_groups:
                    decorr_time_feature = self.compute_decorrelation_time_feature(window_channel)
                    window_channel_features.append(decorr_time_feature)
                window_features.append(window_channel_features)
            dataset_features.append(window_features)
        
        return np.array(dataset_features)
                    
    '''Statistical Moments Features'''
    def compute_statistical_features(self,signal):
        mean = np.mean(signal)
        variance = np.var(signal)
        skewness = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)
        mean_intensity_normalised = np.mean(np.abs(signal))/np.max(np.abs(signal))
        
        return [mean,variance,skewness,kurtosis,mean_intensity_normalised]
    
    '''Spectral Band Features'''
    def compute_spectral_bands_features(self,signal,fs=256):
        freqs, psd = scipy.signal.periodogram(signal,fs,window='hann',scaling='spectrum')
        delta_band_powers = psd[((freqs>=0.5) & (freqs<4))]
        delta_band_power = scipy.integrate.simps(delta_band_powers)
        
        theta_band_powers = psd[((freqs>=4) & (freqs<8))]
        theta_band_power = scipy.integrate.simps(theta_band_powers)
        
        alpha_band_powers = psd[((freqs>=8) & (freqs<13))]
        alpha_band_frequencies = freqs[((freqs>=8) & (freqs<13))]
        alpha_peak_index = np.argmax(alpha_band_powers)
        alpha_peak_freq = alpha_band_frequencies[alpha_peak_index]
        alpha_band_power = scipy.integrate.simps(alpha_band_powers)
        
        beta_band_powers = psd[((freqs>=13) & (freqs<30))]
        beta_band_power = scipy.integrate.simps(beta_band_powers)
        
        gamma_one_band_powers = psd[((freqs>=30) & (freqs<=47))]
        gamma_one_band_power = scipy.integrate.simps(gamma_one_band_powers)
        
        gamma_two_band_powers = psd[((freqs>=53) & (freqs<75))]
        gamma_two_band_power = scipy.integrate.simps(gamma_two_band_powers)
        
        gamma_three_band_powers = psd[((freqs>=75) & (freqs<=90))]
        gamma_three_band_power = scipy.integrate.simps(gamma_three_band_powers)
        
        all_band_powers = [delta_band_power,theta_band_power,alpha_band_power,
                           beta_band_power,gamma_one_band_power,gamma_two_band_power,
                           gamma_three_band_power]
        
        nr_bands = len(all_band_powers)
        all_band_combinations = list(itertools.combinations(np.arange(0,nr_bands),2))
        
        all_band_ratios = []
        
        for first_band,second_band in all_band_combinations:
            bands_ratio = all_band_powers[first_band]/all_band_powers[second_band]
            all_band_ratios.append(bands_ratio)
        
        total_band_power = np.sum(all_band_powers)
        
        relative_delta_band_power = delta_band_power/total_band_power
        relative_theta_band_power = theta_band_power/total_band_power
        relative_alpha_band_power = alpha_band_power/total_band_power
        relative_beta_band_power = beta_band_power/total_band_power
        relative_gamma_one_band_power = gamma_one_band_power/total_band_power
        relative_gamma_two_band_power = gamma_two_band_power/total_band_power
        relative_gamma_three_band_power = gamma_three_band_power/total_band_power
        
        spectral_features = [total_band_power,delta_band_power,theta_band_power,alpha_band_power,beta_band_power,
                             gamma_one_band_power,gamma_two_band_power,gamma_three_band_power,
                             relative_delta_band_power,relative_theta_band_power,relative_alpha_band_power,
                             relative_beta_band_power,relative_gamma_one_band_power,relative_gamma_two_band_power,
                             relative_gamma_three_band_power,alpha_peak_freq]
        
        spectral_features.extend(all_band_ratios)
        
        return spectral_features
    
    '''Compute Spectral Edge Frequency and Spectral Edge Power'''
    def compute_spectral_edge_frequency_features(self,signal,fs=256):
        freqs, power = scipy.signal.periodogram(signal,fs,window='hann',scaling='spectrum')
        power_cum = scipy.integrate.cumtrapz(power)
        sef_50_idx = (np.abs(power_cum - 0.5*scipy.integrate.trapz(power))).argmin() # closest freq holding 50% spectral power
        sef_75_idx = (np.abs(power_cum - 0.75*scipy.integrate.trapz(power))).argmin() # closest freq holding 75% spectral power
        sef_90_idx = (np.abs(power_cum - 0.9*scipy.integrate.trapz(power))).argmin() # closest freq holding 90% spectral power
        sef_50 = freqs[sef_50_idx]
        sef_75 = freqs[sef_75_idx]
        sef_90 = freqs[sef_90_idx]
        sep_50 = power_cum[sef_50_idx]
        sep_75 = power_cum[sef_75_idx]
        sep_90 = power_cum[sef_90_idx]
        
        return [sep_50,sep_75,sep_90,sef_50,sef_75,sef_90]
    
    '''Hjorth Parameters: 
    Activity->Variance(Signal)
    Mobility->Sqrt(Variance(First Derivative)/Variance(Signal))
    Complexity->Mobility(First Derivative)/Mobility(Signal)'''
    def compute_hjorth_parameters_features(self,signal):
        first_deriv_signal = np.diff(signal)
        second_deriv_signal = np.diff(signal,2)
    
        variance_signal = np.mean(signal ** 2)
        variance_first_deriv_signal = np.mean(first_deriv_signal ** 2)
        variance_second_deriv_signal = np.mean(second_deriv_signal ** 2)
    
        activity = variance_signal
        mobility = np.sqrt(variance_first_deriv_signal / variance_signal)
        complexity = np.sqrt(variance_second_deriv_signal / variance_first_deriv_signal) / mobility
    
        return [activity, mobility, complexity]
    
    '''Wavelet Features'''
    def compute_wavelet_features(self,signal,mother_wavelet='db4',level=5):
        coeffs = pywt.wavedec(signal, mother_wavelet, level)
        
        d1_energy = np.sum(np.power(coeffs[-1], 2))
        d2_energy = np.sum(np.power(coeffs[-2], 2))
        d3_energy = np.sum(np.power(coeffs[-3], 2))
        d4_energy = np.sum(np.power(coeffs[-4], 2))
        d5_energy = np.sum(np.power(coeffs[-5], 2))
        a5_energy = np.sum(np.power(coeffs[-6], 2))
        
        d1_energy/=len(coeffs[-1])
        d2_energy/=len(coeffs[-2])
        d3_energy/=len(coeffs[-3])
        d4_energy/=len(coeffs[-4])
        d5_energy/=len(coeffs[-5])
        a5_energy/=len(coeffs[-6])
        
        return [d1_energy,d2_energy,d3_energy,d4_energy,d5_energy,a5_energy]

    '''Decorrelation Time'''
    def compute_decorrelation_time_feature(self,signal):
        xcorr = scipy.signal.correlate(signal,signal)
        autocorr = xcorr[xcorr.size//2:]
        decorr_time = np.where(autocorr<=0)[0][0]
        return decorr_time
