import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print('\n-------------------------------------------------------------------\n')

# Load the dataset
fashion_mnist = keras.datasets.fashion_mnist

# split into tetsing and training data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('train_images.shape : ', train_images.shape) # 60,000 images that are made up of 28x28 pixels (784 in total)
print('test_images.shape : ', test_images.shape)
print('Value of 1 pixel : ', train_images[0,23,23]) # pixel values are between 0 and 255, 0 being black and 255 being white.

# Label values
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizing images
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.show()

# Data Preprocessing
# Applying some prior transformations to our data before feeding it the model
# In this case - scale all our greyscale pixel values (0-255) to be between 0 and 1
# We do this because smaller values will make it easier for the model to process our values.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
print('\nModeling The Neural Network..')
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(len(class_names), activation='softmax') # output layer (3)
])

# Layer 1 - The flatten means the layer will reshape the shape (28,28) array into a vector of 784 neurons so that each pixel will be associated with one neuron
# Flattening the input using flatten() method, (shape [None, 28, 28] is converted to shape [None, 784]
# Layer 2 - Dense denotes each neuron from the previous layer connects to each neuron of this layer, it has 128 neurons and uses the rectify linear unit activation function.
# Layer 3 - It has 10 neurons that we will look at to determine our models output. Each neuron represnts the probabillity of a given image being one of the 10 different classes.
# The activation function softmax is used on this layer to calculate a probabillity distribution for each class
# This means the value of any neuron in this layer will be between 0 and 1, where 1 represents a high probabillity of the image being that class
# Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid, but it also normalizes the sum of the values(output vector) to be 1

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
print('\nTraining the model :')
model.fit(train_images, train_labels, epochs=50)
print('Training Done!')

# Save the model
model.save('models/fashion_mnist.h5')
print('\nModel Saved')
