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
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# model.save("cifar_classification.h5")

# Evaluate the model
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy :', test_acc)
print('Test loss : ', test_loss)

# Making Predictions for all test images
predictions = model.predict(test_images)
print('\nPrediction of 1st image :\n',predictions[0]) # Returns an array of probabillities for each classes
print('\nPredicted class of 1st image : ',class_names[np.argmax(predictions[0])])
print('True value of 1st image : ', class_names[test_labels[0][0]])

predicted_class = map(lambda x : class_names[np.argmax(x)], predictions)
true_class = map(lambda x : class_names[x[0]], test_labels)

df_pred = pd.DataFrame({'True': list(true_class), 'Predicted': list(predicted_class)})
print('\nTrue VS Predicted DataFrame :\n', df_pred.head(10))


# Predicting for singe image
# COLOR = 'white'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label, num):
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  print(f'\nImage-{num}:\nExcpected : {class_names[correct_label]}\nPredicted : {predicted_class}')
  show_image(image, class_names[correct_label], predicted_class, num)

def show_image(img, label, pred, num):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title(f'Image-{num}\nExcpected : {label}')
  plt.xlabel(f'Predicted : {pred}')
  plt.xticks([])
  plt.yticks([])
  plt.colorbar()
  plt.show()

def get_number():
  while True:
    num = input(f'Pick a number between 0 and {test_images.shape[0]}: ')
    if num.isdigit():
      num = int(num)
      if 0 <= num < test_images.shape[0]:
        return int(num)
    else:
      print("Try again...")

print('\nPredicting for single image:')
num = get_number()
image = test_images[num]
label = test_labels[num][0]
predict(model, image, label, num)

is_continue = 'y'
while(is_continue == 'y'):
    is_continue = input('Do you want to continue? [y/n] : ')
    if(is_continue == 'y'):
        num = get_number()
        image = test_images[num]
        label = test_labels[num][0]
        predict(model, image, label, num)
    elif(is_continue == 'n'):
        print("Seems like you've seen enough")
    else:
        print('Oops! Provide a valid input')
        is_continue = 'y'
