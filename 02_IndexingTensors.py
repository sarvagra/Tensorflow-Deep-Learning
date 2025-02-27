# %% importing libraries
import tensorflow as tf
import numpy as np

# %% creating a tensor 
test_tensor=tf.constant([0,1,2,3,4,5,6,7,8,9])
test_tensor1=tf.constant([[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,1]])
test_tensor2=tf.constant([[[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,1]],
                          [[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,2]],
                          [[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,2]]])

# %% using indexing to focus on specific values 1-D
print(test_tensor)
print(test_tensor[0:5]) 
print(test_tensor[5:9])
print(test_tensor[0:10:2])

# %% using indexing to focus on specific values 2-D
print(test_tensor1[1,:])
print(test_tensor1[...,1]) #... means from all rows 
print(test_tensor1[...,0]) 



# %%  using indexing to focus on specific values 3-D
print(test_tensor2[1:, : , : ])
print(test_tensor2[:,1:,:])
print(test_tensor2[:,:,1:]) 
# %%
