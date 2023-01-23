#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)


# # Tensors

# In[2]:


#Scalar
scalar = torch.tensor(5)
scalar


# In[3]:


scalar.ndim #dimensions of the tensor


# In[5]:


scalar.item() 


# In[6]:


# Vector
vector = torch.tensor([7, 7, 8])
vector


# In[7]:


vector.ndim


# In[8]:


vector.shape # size of the tensor


# In[12]:


# Matrix
matrix = torch.tensor([[2, 4, 1], 
                       [6, 8, 1]])
matrix


# In[13]:


matrix.ndim


# In[14]:


matrix.shape


# In[23]:


# Tensor
tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5],
                        [1, 2, 3]]])
tensor


# In[19]:


tensor.ndim


# In[20]:


tensor.shape


# # Random tensor

# In[24]:


random_tensor = torch.rand(size=(2, 3))
random_tensor, random_tensor.dtype


# In[25]:


# Creating a tensor
some_tensor = torch.rand(5, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU


# In[28]:


ones = torch.ones(size=(4, 3))
ones, ones.dtype


# # Zeros and Ones

# In[36]:


zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten


# In[37]:


ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
ten_zeros


# # Tensor datatypes

# In[67]:


# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device= cuda0, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations perfromed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device


# In[45]:


float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

float_16_tensor.dtype


# # Basic operations

# In[46]:


tensor = torch.tensor([1, 3, 5])
tensor2 = torch.tensor([2, 4, 6])


# In[50]:


tensor / 10


# In[51]:


tensor


# In[52]:


tensor = torch.tensor([1, 2, 3]) + 10


# In[53]:


tensor


# In[58]:


tensor + tensor2


# In[59]:


# Element-wise matrix multiplication
tensor * tensor


# In[60]:


# Matrix multiplication
torch.matmul(tensor, tensor)


# In[61]:


# using the "@" symbol 
tensor @ tensor


# # Matrix multiplication

# In[62]:


# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)


# In[63]:


torch.matmul(tensor_A, tensor_B)


# In[64]:


# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)


# In[65]:


output = torch.matmul(tensor_A, tensor_B.T)
output


# # Cuda for GPU

# In[82]:


torch.cuda.is_available()


# In[84]:


float_32_tensor_gpu = torch.cuda.FloatTensor([3.0, 6.0, 9.0]) 

float_32_tensor_gpu.device


# # min, max, mean, sum

# In[85]:


x = torch.arange(0, 100, 10)
x


# In[ ]:


x.min() # .max  .sum


# In[89]:


x.type(torch.float32).mean()


# # changing datatype

# In[91]:


x1_float32=x.type(torch.float32)


# In[92]:


x1_float32.dtype


# In[94]:


x2_float16=x.type(torch.float16)
x2_float16


# In[95]:


x2_float16.dtype


# In[96]:


x


# In[97]:


x.shape


# In[101]:


x_reshaped = x.reshape(1,10)
x_reshaped.shape


# In[109]:


x_stacked = torch.stack([x, x, x], dim=1)
x_stacked


# In[110]:


x_squeezed = x_reshaped.squeeze()


# In[111]:


x_squeezed.shape


# # Indexing

# In[112]:


x = torch.arange (1,10).reshape(1,3,3)
x, x.shape


# In[132]:


x[:, 1, 1]


# In[133]:


import numpy as np


# In[135]:


array = np.arange(1.0, 11.0)
tensor = torch.from_numpy(array)
array, tensor


# In[136]:


tensor2 = torch.from_numpy(array).type(torch.float32)


# In[138]:


tensor2.dtype


# In[139]:


array.dtype


# In[140]:


numpy_tensor = tensor2.numpy()


# In[144]:


numpy_tensor

