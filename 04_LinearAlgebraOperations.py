# %% importing packages
import tensorflow as tf
import numpy as np

# %% creating  tensors
test_tensor=tf.constant([0,1,2])
test_tensor1=tf.constant([[0,1,2],
                          [4,5,7],
                          [5,6,2]], dtype=tf.float16)
test_tensor2=tf.constant([[[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,1]],
                          [[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,2]],
                          [[0,1,2,3,4],
                          [4,5,7,8,9],
                          [5,6,2,0,2]]])
# %%  Matrix multiplication 
mat_prod=tf.linalg.matmul(tf.reshape(test_tensor,(1,3)),test_tensor1)
print(tf.reshape(test_tensor,(1,3)))
print(mat_prod)

# %% Matrix transpose 
print(tf.transpose(test_tensor1))

# %% band part function
print(tf.linalg.band_part(test_tensor1,0,0))  ## output is always diagonal for a 0,0 
print(tf.linalg.band_part(test_tensor1,0,-1))  ## output is always upper triangular for a 0,0 
print(tf.linalg.band_part(test_tensor1,-1,0))  ## output is always lower triangular for a 0,0 


# %% find the inverse
print(tf.linalg.inv(test_tensor1))

# %% einsum transpose
tp=np.array([[0,1,2],
            [4,5,7],
            [5,6,2]])
print(tf.einsum("ij -> ji",tp))

# %%
