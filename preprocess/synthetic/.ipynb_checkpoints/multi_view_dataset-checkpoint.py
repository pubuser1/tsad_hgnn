#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:20:27 2023

@author: vinitha
"""

import torch
import numpy as np
from scipy import signal 

# Set random seed for reproducibility
np.random.seed(42)

# Time values
time_axis = np.linspace(0, 2 * np.pi, 1000)

#-------------------------- Dataset 1 ------------------------------------------
def get_dataset():
   sine_waves = []
   for frequency in range(1,6):
      sine_wave = np.sin(frequency * time_axis)
      sine_waves.append(sine_wave)
   sine_waves = np.vstack(sine_waves)
     
   cos_waves = []
   for frequency in range(1,6):
      cos_wave = np.cos(frequency * time_axis)
      cos_waves.append(cos_wave)
   cos_waves = np.vstack(cos_waves)

   sum_waves = []
   for frequency in range(1,6):
      sine_wave = np.sin(frequency * time_axis)
      cos_wave = np.cos(frequency * time_axis)
     
      sum_wave = []
      for sine, cos in zip(sine_wave,cos_wave):
         sum_wave.extend([sine+cos])
         
      sum_waves.append(sum_wave)
   sum_waves = np.vstack(sum_waves)
     
   square_waves1 = []
   for frequency in range(1,5):
      square_wave = frequency + np.sign(np.sin(2 * np.pi * frequency * time_axis))
      square_waves1.append(square_wave)
   square_waves1 = np.vstack(square_waves1)
     
   square_waves2 = []
   for frequency in range(1,5):
      square_wave = frequency * np.sign(np.cos(2 * np.pi * frequency * time_axis))
      square_waves2.append(square_wave)
   square_waves2 = np.vstack(square_waves2)
      
   return [sine_waves, cos_waves, sum_waves, square_waves1, square_waves2]
 
#-------------------------- Dataset 2 ------------------------------------------  

def get_dataset2():
  
  #------ View 1 -------------
  sine_wave1 = np.sin(time_axis)
  sine_wave2 = 0.5 * np.sin(time_axis)
  saw_tooth1 = signal.sawtooth(2 * np.pi * 3 * time_axis, width = 1)
  view_1 = [sine_wave1, sine_wave2, saw_tooth1]
  
  #------ View 2 -------------
  square_wave1 = 1 + np.sign(np.sin(2 * np.pi * time_axis))
  square_wave2 = np.sign(np.sin(2 * np.pi * time_axis))
  triangle1 = signal.sawtooth(2 * np.pi * 4 * time_axis, width = 0.5)
  view_2 = [square_wave1, square_wave2, triangle1]
  
  #------ View 3 -------------
  cos_wave1 = np.cos(time_axis)
  #sinc_wave3 = 0.5 + np.cos(time)  
  cos_wave2 = 0.5 * np.cos(time_axis)
  sinc_wave1 = np.sinc(time_axis) + np.sin(time_axis) + np.cos(time_axis)
  view_3 = [cos_wave1, cos_wave2, sinc_wave1]
  
  A =  np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
  
  num_periods = 1000
  num_variables = 5

  data = np.zeros((num_variables,num_periods))
  
  data[0,0]=0
  data[1,0]=1
  data[2,0]=2
  data[3,0]=-2
  data[4,0]=-1

  for t in range(1, num_periods):  
   data[:,t] = np.matmul(A,data[:,t-1]) 
   mask = np.abs(data[:,t]) > 5 
   data[:,t] = data[:,t] - 2 * np.multiply(mask , np.sign(data[:,t])*(np.abs(data[:,t]) - 5))
   
  view_4 = data[:3, :]
  view_5 = data[3:, :]
  
  return [view_1, view_2, view_3, view_4, view_5]

#-------------------------- Dataset 3 ------------------------------------------  

def get_dataset3():
  #------ View 1 -------------
  sine_wave1 = np.sin(time_axis)
  sine_wave2 = 0.5 * np.sin(time_axis)
  saw_tooth1 = signal.sawtooth(2 * np.pi * 3 * time_axis, width = 1)
  #view_1 = [sine_wave1, sine_wave2, saw_tooth1]
  view_1 = np.vstack((sine_wave1, sine_wave2, saw_tooth1))
  
  #------ View 2 -------------
  square_wave1 = 1 + np.sign(np.sin(2 * np.pi * time_axis))
  square_wave2 = np.sign(np.sin(2 * np.pi * time_axis))
  triangle1 = signal.sawtooth(2 * np.pi * 4 * time_axis, width = 0.5)
  #view_2 = [square_wave1, square_wave2, triangle1]
  view_2 = np.vstack((square_wave1, square_wave2, triangle1))
  
  #------ View 3 -------------
  cos_wave1 = np.cos(time_axis)
  #sinc_wave3 = 0.5 + np.cos(time)  
  cos_wave2 = 0.5 * np.cos(time_axis)
  sinc_wave1 = np.sinc(time_axis) + np.sin(time_axis) + np.cos(time_axis)
  #view_3 = [cos_wave1, cos_wave2, sinc_wave1]
  view_3 = np.vstack((cos_wave1, cos_wave2, sinc_wave1))
  
  A =  np.array([[0, 0, 1, 0, 0], [0, 0.6, 0.4, 0, 0], [0.2, 0.35, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0.5, 0.5]])

  num_periods = 1000
  num_variables = 5

  data = np.zeros((num_variables,num_periods))

  data[:,0] = np.random.uniform(-5, 5, num_variables)

  for t in range(1, num_periods):
   data[:,t] = np.matmul(A,data[:,t-1]) + np.random.randn(num_variables)*0.2
   
  #view_4 = data[:3, :]
  #view_5 = data[3:, :]
  
  #return [view_1, view_2, view_3, view_4, view_5]
  return [view_1, view_2, view_3, data]

#-------------------------- Dataset 4 ------------------------------------------    

def generate_multivariate_time_series(covariance_matrix, num_samples):
    dim = covariance_matrix.shape[0]
    mean = torch.zeros(dim)
    multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix)
    samples = multivariate_normal.sample((num_samples,))
    return samples

def generate_block_diagonal_covariance(num_variables, group_sizes, correlations):
    covariance_matrix = torch.zeros(num_variables, num_variables)

    # Populate the diagonal blocks
    start_index = 0
    for size, correlation in zip(group_sizes, correlations):
        end_index = start_index + size

        # Set the correlation within the group
        shape = (size, size)
        block = torch.rand(shape)
        #block = torch.randn(size, size)
        block = torch.mm(block, block.t())  # Ensure positive definite
        block = correlation * block
        covariance_matrix[start_index:end_index, start_index:end_index] = block

        start_index = end_index

    return covariance_matrix  

def get_dataset4():
    num_variables = 23

    # Group sizes
    group_sizes = [5, 5, 5, 4, 4]
    
    # Correlations within each group
    correlations = [0.9, 0.9, 0.9, 0.9, 0.9]

    # Generate block diagonal covariance matrix
    cov_matrix = generate_block_diagonal_covariance(num_variables, group_sizes, correlations)

    # Number of samples to generate
    num_samples = 100

    # Generate synthetic time series data
    time_series = generate_multivariate_time_series(cov_matrix, num_samples)
    time_series = (time_series.T).float()
    
    data = []
    start = 0
    for group_size in group_sizes:
      end = start + group_size
      data.append(time_series[ start : end,:])
      start = end
      
    return data
  
#--------------------- Multi View Data comprising of sinusoidal signals ---------------
#Pi_Si : Person i Signal i
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
    P1_S1 = (amp_1*np.sin(2 * np.pi * f_1 * x)) + noise_1
    P1_S2 = (amp_2*np.sin(2 * np.pi * f_1 * x)) + noise_1

    P2_S1 = (amp_1*np.sin(2 * np.pi * f_2 * x)) + noise_2
    P2_S2 = (amp_2*np.sin(2 * np.pi * f_2 * x)) + noise_2
    
    data1 = np.vstack([P1_S1[0], P1_S2[0], P2_S1[0], P2_S2[0]])
    data2 = np.vstack([P1_S1[10], P1_S2[10], P2_S1[10], P2_S2[10]])
    
    return data1,data2
  
def Sine_With_Freq_Change():
    # Parameters
    f_1 = np.append([0.01]*400, [0.02]*400)
    f_2 = np.append([0.005]*400, [0.01]*400)
    sample = 800
    n_samples = 1024

    # Amplitudes
    amp_1 = np.random.normal(1, 0.05, n_samples)
    amp_2 = np.random.normal(1.5, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    P1_S1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    P1_S2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    P2_S1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    P2_S2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2
    
    data1 = np.vstack([P1_S1[0], P1_S2[0], P2_S1[0], P2_S2[0]])
    data2 = np.vstack([P1_S1[10], P1_S2[10], P2_S1[10], P2_S2[10]])
    
    return data1,data2
  
#def multi_view_sines():
    #view1, view2 = Sine()
    #view3, view4 = Sine_With_Freq_Change()
    
    #return [view1, view2, view3, view4]
  
#--------------------- Multi View Data comprising of sinusoidal signals ---------------
def Sine_View1_View2():
  
    n_samples = 1000
    
    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_1 = amp_1.reshape(-1,1)
    amp_2 = np.random.normal(0.6, 0.05, n_samples) 
    amp_2 = amp_2.reshape(-1,1)
    
    #------------------------ View 1 -----------------------------
    # Parameters
    f_1 = 0.01
    f_2 = 0.005
    sample = 800  
    
    # Generate noise
    x1 = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))   

    # Generate signal
    V1_X1 = (amp_1*np.sin(2 * np.pi * f_1 * x1)) + noise_1
    V1_X2 = (amp_2*np.sin(2 * np.pi * f_1 * x1)) + noise_1

    V1_X3 = (amp_1*np.sin(2 * np.pi * f_2 * x1)) + noise_2
    V1_X4 = (amp_2*np.sin(2 * np.pi * f_2 * x1)) + noise_2
    
    view1 = np.vstack([V1_X1[0], V1_X2[0], V1_X3[0], V1_X4[0]])
    print('Set 1 Data 1 : ',view1.shape)
    
    #------------------------ View 2 -----------------------------
    # Parameters
    f_1 = 0.005
    f_2 = 0.0025
    sample = 800 #1600
    
    # Generate noise
    x2 = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))   

    # Generate signal
    V2_X1 = (amp_1*np.sin(2 * np.pi * f_1 * x2)) + noise_1
    V2_X2 = (amp_2*np.sin(2 * np.pi * f_1 * x2)) + noise_1

    V2_X3 = (amp_1*np.sin(2 * np.pi * f_2 * x2)) + noise_2
    V2_X4 = (amp_2*np.sin(2 * np.pi * f_2 * x2)) + noise_2
    
    view2 = np.vstack([V2_X1[10], V2_X2[10], V2_X3[10], V2_X4[10]])
    print('Set 1 Data 2 : ',view2.shape)
    
    #-------------- Return both the views of data ---------------------
    return view1,view2
  
def Sine_View3_View4():
  
    n_samples = 1024  

    # Amplitudes
    amp_1 = np.random.normal(1, 0.05, n_samples)
    amp_2 = np.random.normal(1.5, 0.05, n_samples)

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)
    
    #------------------------ View 3 -----------------------------
    # Parameters
    f_1 = np.append([0.01]*400, [0.02]*400)
    f_2 = np.append([0.005]*400, [0.01]*400)
    sample = 800  

    # Generate noise
    x1 = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    # Generate signal
    V3_X1 = amp_1*np.sin(2 * np.pi * f_1 * x1) + noise_1
    V3_X2 = amp_2*np.sin(2 * np.pi * f_1 * x1) + noise_1

    V3_X3 = amp_1*np.sin(2 * np.pi * f_2 * x1)+ noise_2
    V3_X4 = amp_2*np.sin(2 * np.pi * f_2 * x1) + noise_2
    
    view3 = np.vstack([V3_X1[0], V3_X2[0], V3_X3[0], V3_X4[0]])
    print('Set 2 Data 1 : ',view3.shape)
    
    #------------------------ View 4 -----------------------------
    # Parameters
    f_1 = np.append([0.005]*400, [0.01]*400)
    f_2 = np.append([0.0025]*400, [0.005]*400)
    #f_1 = np.append([0.005]*800, [0.01]*800)
    #f_2 = np.append([0.0025]*800, [0.005]*800)
    sample = 800 #1600  

    # Generate noise
    x2 = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    # Generate signal
    V4_X1 = amp_1*np.sin(2 * np.pi * f_1 * x2) + noise_1
    V4_X2 = amp_2*np.sin(2 * np.pi * f_1 * x2) + noise_1

    V4_X3 = amp_1*np.sin(2 * np.pi * f_2 * x2)+ noise_2
    V4_X4 = amp_2*np.sin(2 * np.pi * f_2 * x2) + noise_2
    
    view4 = np.vstack([V4_X1[10], V4_X2[10], V4_X3[10], V4_X4[10]])
    print('Set 2 Data 2 : ',view4.shape)
    
    #-------------- Return both the views of data ---------------------
    return view3,view4
  
def Sine_multi_view():
    view1, view2 = Sine_View1_View2()
    view3, view4 = Sine_View3_View4()
    
    return [view1, view2, view3, view4]