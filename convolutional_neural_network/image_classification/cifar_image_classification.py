# Trying to classify 10 different everyday objects
# The labels in this dataset are the following:
#     Airplane
#     Automobile
#     Bird
#     Cat
#     Deer
#     Dog
#     Frog
#     Horse
#     Ship
#     Truck

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

print('\n-------------------------------------------------------------------\n')

# Load Dataset, split into traing set and testing set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Label values
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('train_images.shape : ', train_images.shape)
print('test_images.shape : ', test_images.shape)
print('train_labels[0] : ', train_labels[0])
print('Class label of 1st train label : ', class_names[train_labels[0][0]])

# Visualizing 1st 25 images with their labels
plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]], fontsize=5)
plt.tight_layout()
plt.savefig('plots/classes.png')
# plt.show()

# CNN Architecture
# Create the convolutional base
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size
# configure CNN to process inputs of shape (32, 32, 3), by passing the argument input_shape

# Arguements:
# filter -  the number of output filters in the convolution
# kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
# pool_size - integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.summary()
print('Convolution base created successfully..\n')
# The width and height dimensions tend to shrink as you go deeper in the network.
# The number of output channels for each Conv2D layer is controlled by the first argument-filters
# Typically, as the width and height shrink, you can afford, computationally, to add more output channels in each Conv2D layer

# Adding Dense layer
# feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into Dense layers to perform classification
# Dense layers take 1D vectors as input, but current output as 3D
# So use Flatten layer, then add Dense layer, then add final Dense layer with 10 output classes

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names)))

model.summary()
print('Model created successfully...\n')

# Compile and train the model
print('Compiling the model..')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('Training the model..\n')
history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

# Save the model
model.save("models/cifar_classification.h5")
print('\nModel Saved...')

# Learning curves
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('plots/epoch_vs_accuracy.png')
plt.show()
