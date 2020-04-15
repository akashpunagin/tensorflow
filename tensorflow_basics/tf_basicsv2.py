import tensorflow as tf

print('\n---------------------------------------\n')
print('Tensorflow Version : ', tf.version)


# Creating Tensors
# Variables
string_type = tf.Variable('Hello World', tf.string)
int_type = tf.Variable(42, tf.int16)
float_type = tf.Variable(32.11, tf.float32)

print('String : ',string_type)
print('Int : ',int_type)
print('Float : ',float_type)

# Rank 1, Rank 2, Rank 3 Tensor (Dimention)
rank_0 = tf.Variable(23, tf.int32) # Scalar Vecrtor
rank_1 = tf.Variable(['hi', 'hello', 'good'], tf.string)
rank_2 = tf.Variable([[1,2,3],[4,5,6],[7,8,9]], tf.int32)

print('\nRank : ')
print('Rank 0 : ',tf.rank(rank_0))
print('Rank 1 : ',tf.rank(rank_1))
print('Rank 2 : ',tf.rank(rank_2))

# Shape
print('\nShape : ')
print('Rank 0 : ',rank_0.shape)
print('Rank 1 : ',rank_1.shape)
print('Rank 2 : ',rank_2.shape)

# Changing shape
tensor_1 = tf.ones([2,2,3])
tensor_2 = tf.reshape(tensor_1, [3,2,2])
tensor_3 = tf.reshape(tensor_1, [4, -1])


print('Original :\n', tensor_1)
print('Reshape:\n', tensor_2)
print('Reshape:\n', tensor_3)

# Types of Tensors
# 1. Variable
# 2. Constant
# 3. Placeholder
# 4. Sparse Matrix
