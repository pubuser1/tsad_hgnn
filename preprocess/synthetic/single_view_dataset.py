#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:22:36 2023

@author: vinitha
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def Sine():
    # Parameters
    f_1 = 0.01
    f_2 = 0.005
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = (amp_1*np.sin(2 * np.pi * f_1 * x)) + noise_1
    sin_1_person_2 = (amp_2*np.sin(2 * np.pi * f_1 * x)) + noise_1

    sin_2_person_1 = (amp_1*np.sin(2 * np.pi * f_2 * x)) + noise_2
    sin_2_person_2 = (amp_2*np.sin(2 * np.pi * f_2 * x)) + noise_2
    
    data = np.vstack([sin_1_person_1[0], sin_1_person_2[0], sin_2_person_1[0], sin_2_person_2[0]])
    
    return data
  
def Sine_With_Freq_Change():
    # Parameters
    f_1 = np.append([0.01]*400, [0.02]*400)
    f_2 = np.append([0.005]*400, [0.01]*400)
    sample = 800
    n_samples = 1024

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2
    
    data = np.vstack([sin_1_person_1[0], sin_1_person_2[0], sin_2_person_1[0], sin_2_person_2[0]])
    
    return data
  
def Sine_With_Adv_Freq_Change():
    # Parameters
    f_1 = np.array([0.01]*200 + [0.02]*400 + [0.01]*200)
    f_2 = np.array([0.005]*200 + [0.01]*400 + [0.005]*200)
    sample = 800
    n_samples = 1024

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2
    
    data = np.vstack([sin_1_person_1[0], sin_1_person_2[0], sin_2_person_1[0], sin_2_person_2[0]])
    
    return data

def Sine_With_Shift():  
    # Parameters
    f_1 = 0.01
    f_2 = 0.005
    sample = 800
    n_samples = 1024

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))
    noise_shift_1 = np.random.normal(0,10,n_samples).reshape(-1,1)
    noise_shift_2 = np.random.normal(0,10,n_samples).reshape(-1,1)

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = (amp_1*np.sin(2 * np.pi * f_1 * x + noise_shift_1)) + noise_1
    sin_1_person_2 = (amp_2*np.sin(2 * np.pi * f_1 * x + noise_shift_1)) + noise_1

    sin_2_person_1 = (amp_1*np.sin(2 * np.pi * f_2 * x + noise_shift_2)) + noise_2
    sin_2_person_2 = (amp_2*np.sin(2 * np.pi * f_2 * x + noise_shift_2)) + noise_2

    data = np.vstack([sin_1_person_1[0], sin_1_person_2[0], sin_2_person_1[0], sin_2_person_2[0]])
    
    return data