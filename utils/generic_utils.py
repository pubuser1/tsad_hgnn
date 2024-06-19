
import numpy as np
import os

import torch
import torch.nn.functional as F

#------------------------ Initialize a square symmetric matrix of given shape -----------------------
def initial_adj_matrix_1(shape):
  arr = np.random.uniform(0, 1, size=(shape,shape))
  x = (torch.from_numpy(arr.astype(np.float32))) 
  return x
  #np.fill_diagonal(arr, 0)
  #arr = np.random.uniform(-1, 1, size=(shape,shape))
  #arr = np.zeros((shape,shape))
  #return arr

#------------------------ Initialize a square symmetric matrix of given shape 
#----------------------- And with diagonal elements zero ---------
def initial_adj_matrix_2(shape):
  array_shape = (shape,shape)
  x_np = np.random.uniform(0, 1, size=array_shape)
  #x = (torch.from_numpy(x_np.astype(np.float32))) 
  x = torch.from_numpy(x_np).double()

  for i in range(shape):
   x[i, i] = float('-inf')

  softmax_x = F.softmax(x, dim=1)
  #print(softmax_x)
  return softmax_x

#------------------------ Initialize a square symmetric torch tensor of given shape ---------
def initial_adj_matrix_3(shape):
  array_shape = (shape,shape)
  
  #x_np = np.random.uniform(0, 1, size=array_shape)
  x_np = np.random.uniform(0, 1, size=array_shape)
  x = torch.from_numpy(x_np).double()
  
  #print(x)

  return x

#------------------------ Initialize a square symmetric torch tensor of given shape ---------
def initial_adj_matrix_4(shape):
  array_shape = (shape,shape)
  
  x_np = np.full(array_shape, 1/shape)
  np.fill_diagonal(x_np, 0)
  
  x = torch.from_numpy(x_np).double()
  
  #print(x)

  return x


#------------------------ Initialize a square symmetric torch tensor of given shape ---------
def initial_adj_matrix(shape):
  return torch.rand(shape,shape)

#------------------------ Initialize a square zero torch tensor of given shape ---------
def initialize_zero_tensor(shape):
    tensor = torch.zeros(shape, shape)  # Initialize with zeros
    return tensor

#------------------------ Initialize a matrix of given shape -----------------------
def initialize_transform(shape1, shape2):
  array_shape = (shape1,shape2)
  x_np = np.random.uniform(0, 1, size=array_shape)
  #x = (torch.from_numpy(x_np.astype(np.float32))) 
  x = torch.from_numpy(x_np).double()
  #print(x)
  #return x
  softmax_x = F.softmax(x, dim=1)
  #print(softmax_x)
  return softmax_x

#------------- Create directory to save results ---------------------------------------------
def create_directory(directory):
 parent_dir = os.getcwd()
 path = os.path.join(parent_dir, directory)
 
 try:
  os.mkdir(path)
  #print("Directory '%s' created" % directory)
 except FileExistsError:
  print('')

#------------------------ Regenerate Data From Embeddings -----------------------
def regenerate_data(Embeddings):

 num_vars = Embeddings.shape[0]
 num_sequences = Embeddings.shape[1]
 window_size = Embeddings.shape[2]

 data_length = num_sequences + window_size - 1

 regen_data = np.zeros((num_vars, data_length))

 for var in range(num_vars) :
  I1 = 0
  I2 = 0
  num_iterations = 0

  seqData = Embeddings[var]
  varData = np.zeros(data_length)

  for index in range(data_length):
   sum = 0

   if index < (window_size - 1):
    I1 = index
    I2 = 0
    num_iterations = index + 1

   elif index > (num_sequences - 1):
    I1 = num_sequences - 1
    I2 = index - I1
    num_iterations = window_size - (index - (num_sequences - 1))

   else :
    I1 = index
    I2 = 0
    num_iterations = window_size

   for iter in range(num_iterations):
    sum = sum + seqData[I1, I2]
    I1 = I1 - 1
    I2 = I2 + 1

   varData[index] = sum / num_iterations

  regen_data[var] = varData
  #print(varData)

 return regen_data
