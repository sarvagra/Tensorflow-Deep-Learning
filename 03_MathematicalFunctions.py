# %% importing packages
import tensorflow as tf

# %% Creating tensor
tensor0=tf.constant([[1,-5,9,0,-22],
                     [-3,-5,-69,7,32],
                     [94,2,3,0,-12]
                     ])
tensor1=tf.constant([[2,3,0,-1,-22],
                     [0,1,-5,9,4],
                     [11,2,3,0,-12]
                       ])
tensor2=tf.constant([1,2,7,5,9,34,23,12])
print(tensor0)
# %% abs() function , turns all values to +ve
abs_tensor=tf.math.abs(
    tensor0, name=None
)
print(abs_tensor)

# %% addition 
print(tf.add(tensor0,tensor1))

# %% substraction
print(tf.subtract(tensor0,tensor1))

# %% division
print(tf.divide(tensor0,tensor1))
print(tf.math.divide_no_nan(tensor0,tensor1))

# %% argmax() : returns the index of maximim entry in the tensor
print(tf.math.argmax(tensor2))
print(tf.math.argmax(tensor0)) # returns the index of every greatest element in each column

# %% pow(x,y) : raises each ekement in x to the power of the element at same position in y
tensor0float=tf.cast(tensor0,dtype=tf.float32)
tensor1float=tf.cast(tensor1,dtype=tf.float32)
print(tf.pow(tensor0float,tensor1float))

# %% reduce sum
red_sum=tf.reduce_sum(tensor0)
print(red_sum)  #finds sum of all elements in the tensor
red_max=tf.reduce_max(tensor0) 
print(red_max) # finds the maximum element in the tensor
red_min=tf.reduce_min(tensor0) 
print(red_min) # finds the minimum element in the tensor

