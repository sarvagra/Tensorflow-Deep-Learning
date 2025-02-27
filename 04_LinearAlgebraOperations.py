# %% importing packages
import tensorflow as tf

# %% creating  tensors
test_tensor=tf.constant([0,1,2])
test_tensor1=tf.constant([[0,1,2],
                          [4,5,7],
                          [5,6,2]])
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

# %%
