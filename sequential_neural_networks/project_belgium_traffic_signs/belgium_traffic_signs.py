import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import skimage.io
import os
import random
import tensorflow as tf
from tensorflow import keras

print('\n-------------------------------------------------------------------\n')

# Function for loading data
def load_data(data_dir):
    # Get all subdirectories of data_dir (Each subdirectory represents a label)
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through directories and collect the data in two lists: labels and images
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")] # Portable Pixmap Format
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/akash/Workplace/Tensorflow Workplace/sequential_neural_networks/project_belgium_traffic_signs"
train_data_dir = os.path.join(ROOT_PATH, "dataset/Training")
test_data_dir = os.path.join(ROOT_PATH, "dataset/Testing")

images, labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)
print('\nDataset loaded successfully...\n')

# Convert python array to np.array
images_array = np.array(images)
labels_array = np.array(labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# dimensions of images_array
print('Dimentions of images_array : ', images_array.ndim)

# dimensions of first instance of images_array
print('Dimentions of singe image in images_array : ', images_array[0].ndim)

# number of `images_array's elements
print('Number of elements in images_array : ', images_array.size)

# dimensions of labels_array
print('Dimentions of labels_array : ', labels_array.ndim)

# number of `labels_array's elements
print('Number of elements in labels_array : ', labels_array.size)

# Count the number of labels
print('Number of labels : ', len(set(labels_array)))

print('train_images.shape : ', images_array.shape)
print('test_images.shape : ', test_images.shape)


# Distributions of Traffic Signs
# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, bins=len(set(labels_array)))
plt.tight_layout()
# plt.show()
# Observation - there are labels that are more heavily present in the dataset than others, labels - 22, 32, 38, and 61

# Visualizing The Traffic Signs

# Function to Visualize all unique signs
def display_unique_signs(labels):
    # Get the unique labels
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure()

    # For each unique label,
    for i,label in enumerate(unique_labels):
        # pick the first image for each label
        image = images[labels.index(label)]
        # Define 64 subplots
        plt.subplot(8, 8, i+1)
        # Don't include axes
        plt.axis('off')
        # Add a title to each subplot
        plt.title(f"Label {label} ({labels.count(label)})", fontsize=3)
        plt.tight_layout()
        # And you plot this first image
        plt.imshow(image)

# Display all unique signs
display_unique_signs(labels)
# plt.show()

# Determine the (random) indexes of the images that you want to see
n_img_to_disp = 4
random_index = random.sample(range(0, images_array.size), n_img_to_disp)
print('Random indexes : ', random_index)

# Function to display random images
def display_random_images(images_array, random_index, n_img_to_disp, msg_to_print, cmap_arg='viridis'):
    # Initialize the figure
    plt.figure()

    # Fill out the subplots with the random images that you defined
    print(f'\n{msg_to_print} :')
    for i in range(len(random_index)):
        plt.subplot(1, n_img_to_disp, i+1)
        plt.axis('off') # Don't include axes
        plt.imshow(images_array[random_index[i]], cmap=cmap_arg)
        plt.title(f'shape:{images_array[random_index[i]].shape}', fontsize=6)
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.5)
        print(f'Image-{i+1} - shape: {images_array[random_index[i]].shape}, min: {images_array[random_index[i]].min()}, max: {images_array[random_index[i]].max()}')

# Display random images with random_index
display_random_images(images_array, random_index, n_img_to_disp, 'Before Rescaling')
# plt.show()
# Observation - all images are not the same, size of images are unequal

# Data Preprocessing

# Rescaling images to 20 by 28 pixels
images28 = [transform.resize(image, (28, 28)) for image in images_array]
images28 = np.array(images28)
test_images = [transform.resize(image, (28, 28)) for image in test_images]
test_images = np.array(test_images)
# Display same random images which are rescaled
display_random_images(images28, random_index, n_img_to_disp, 'After Rescaling')
# plt.show()
# Observation - all the images now have the same size

# Image Conversion to Grayscale
images28 = rgb2gray(images28)
test_images = rgb2gray(test_images)
# Display same random images which are converted to greyscale
display_random_images(images28, random_index, 4, 'After converting to Greyscale', cmap_arg='gray')
# plt.show()
# Observation - all the images have the same size


# Deep Learning With TensorFlow

# Building the model
print('\nModeling The Neural Network..')
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(set(labels_array)), activation='softmax') # output layer (3)
])

# Flattening the input using flatten() method, (shape [None, 28, 28] is converted to shape [None, 784]
# Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid, but it also normalizes the sum of the values(output vector) to be 1

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
print('\nTraining the model :')
model.fit(images28, labels_array, epochs=100)
print('Training Done!')

# Save the model
model.save('models/belgium_traffic_signs_model.h5')
print('\nModel Saved')
