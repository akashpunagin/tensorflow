# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

print('\n-------------------------------------------------------------\n')

# Print the result
print('Without Session : ',result)

with tf.compat.v1.Session() as sess:

    # Initialize two constants (Build a dataflow graph)
    x1 = tf.constant([1,2,3,4])
    x2 = tf.constant([5,6,7,8])

    # Multiply
    result = tf.multiply(x1, x2)

    # Print the result with session
    print('With Session : ',sess.run(result))
    print(result.eval())

    # Close the session
    sess.close()

print('\n-------------------------------------------------------------\n')

# Session with log_device_placement - information of which device is used for each operation
# This config tells you which device the operation is allocated while building the graph
# It can always find the prioritized device with best performance on you machine
config=tf.compat.v1.ConfigProto(log_device_placement=True)
with tf.compat.v1.Session(config=config) as sess:

    # Initialize two constants (Build a dataflow graph)
    x1 = tf.constant([1,2,3,4])
    x2 = tf.constant([5,6,7,8])

    # Multiply
    result = tf.multiply(x1, x2)

    # Print the result with session
    print('With Session : ',sess.run(result))

    # Close the session
    sess.close()

print('\n-------------------------------------------------------------\n')

# Session with allow_soft_placement - allows dynamic allocation of GPU memory,
# If allow_soft_placement is true, an operation will be placed on CPU if
#   1. there's no GPU implementation for the operation
#   or
#   2. no GPU devices are known or registered
#   or
#   3. need to co-locate with reftype input(s) which are from CPU.
# If tensorflow is GPU supported, the operations always perform on GPU no matter if allow_soft_placement is set or not, and even if you set device as CPU
# But if you set allow_soft_placement as false and device as GPU but GPU cannot be found in your machine it raises error.
config=tf.compat.v1.ConfigProto(allow_soft_placement=False)
with tf.compat.v1.Session(config=config) as sess:
    tf.device('/gpu')

    # Initialize two constants (Build a dataflow graph)
    x1 = tf.constant([1,2,3,4])
    x2 = tf.constant([5,6,7,8])

    # Multiply
    result = tf.multiply(x1, x2)

    # Print the result with session
    print('With Session : ',sess.run(result))

    # Close the session
    sess.close()
