# %% importing tensorflow
import tensorflow as tf

# %% creating a 0 dim tensor
tensor_zero_d=tf.constant(4)
print(tensor_zero_d)
# %% creating a 1 dim tensor
tensor_one_d=tf.constant([1,2,3,-4])
print(tensor_one_d)
# %% creating a 2 dim tensor
tensor_two_d = tf.constant([
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ])
print(tensor_two_d)
# %% creating a 3 dim tensor 
tensor_three_d = tf.constant([
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],

    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]]
    
    ])
print(tensor_three_d)

# %% get shape of the tensor (r,h,c)
print(tensor_three_d.shape)
# %% creating a 4 dim tensor
tensor_four_d = tf.constant([
    [
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],

    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]]
    
    ],
    [
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],

    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]
    ],
    [
    [1,2,3,4],
    [0,0,0,2],
    [5,8,6,9]]
    
    ]
    
    ])
print(tensor_four_d)
print(tensor_four_d.shape) # (depth,row,column,layers)

# %%  convert an array into tensor
import numpy as np
arr=np.array([1,2,3,4])
print(arr)
conv_arr=tf.convert_to_tensor(arr)
print(conv_arr)


## Eye method

# %% generate identity matrix (using rows)
eye_tensor=tf.eye(
    num_rows=4,
    num_columns=3,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
print(eye_tensor)
print(69*eye_tensor) #multiplies the number to all entries 


# %% using columns
eye_tensor1=tf.eye(
    num_rows=4,
    num_columns=3,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
print(eye_tensor1)

# %% using shape
eye_tensor2=tf.eye(
    num_rows=4,
    num_columns=3,
    batch_shape=[2,2],
    dtype=tf.dtypes.float32,
    name=None
)
print(eye_tensor2)


## Fill method 

# %% creates a tensor filled with scalar value
fill_tensor=tf.fill(
    [2,3], #shape 
     0, # element to be filled 
     name=None
)
print(fill_tensor)


## Ones method

# %% Creates a tensor filled with ones only
ones_tensor=tf.ones(
    shape=[3,3],
    dtype=tf.dtypes.float32,
    name=None,
    layout=None
)
print(ones_tensor)

## Zeros method

# %% Creates a tensor filled with zeros only
ones_tensor=tf.zeros(
    shape=[3,3],
    dtype=tf.dtypes.float32,
    name=None,
    layout=None
)
print(ones_tensor)

## Ones Likemethod

# %% changes all entries to ones while maintaining the shape
like_ones=tf.ones_like(tensor_two_d)
print(like_ones)


## Rank method

# %% finds the rank of a tensor
rank=tf.rank(
    tensor_four_d, name=None
)
print (rank)


## Size method

# %% Returns the size of a tensor.
print(
    tf.size(
    like_ones, out_type=None, name=None
)
)


## Random Normal Tensor method
# %% 
rand_norm=tf.random.normal(
    shape=[4,4],
    mean=100, #all close to this number
    stddev=1.0,
    dtype=tf.dtypes.float64,
    seed=None,
    name=None
)
print(rand_norm)
# %% 
