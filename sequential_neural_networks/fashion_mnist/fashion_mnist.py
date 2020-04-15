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
model.fit(train_images, train_labels, epochs=10)
print('Training Done!')

# Evaluating the model
print('\nEvaluating model for test data :')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('Test accuracy :', test_acc)
print('Test loss : ', test_loss)

# Making Predictions for all test images
predictions = model.predict(test_images)
print('\nPrediction of 1st image :\n',predictions[0]) # Returns an array of probabillities for each classes
print('\nPredicted class of 1st image : ',class_names[np.argmax(predictions[0])])
print('True value of 1st image : ', class_names[test_labels[0]])

predicted_class = map(lambda x : class_names[np.argmax(x)], predictions)
true_class = map(lambda x : class_names[x], test_labels)

df_pred = pd.DataFrame({'True': list(true_class), 'Predicted': list(predicted_class)})
print('\nTrue VS Predicted DataFrame :\n', df_pred.head(10))


# Predicting for singe image
# COLOR = 'white'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label, num):
  # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
  #              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  print(f'\nImage-{num}:\nExcpected : {class_names[correct_label]}\nPredicted : {predicted_class}')
  show_image(image, class_names[correct_label], predicted_class, num)


def show_image(img, label, pred, num):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title(f'Image-{num}\nExcpected : {label}')
  plt.xlabel(f'Predicted : {pred}')
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
label = test_labels[num]
predict(model, image, label, num)

is_continue = 'y'
while(is_continue == 'y'):
    is_continue = input('Do you want to continue? [y/n] : ')
    if(is_continue == 'y'):
        num = get_number()
        image = test_images[num]
        label = test_labels[num]
        predict(model, image, label, num)
    elif(is_continue == 'n'):
        print("Seems like you've seen enough")
    else:
        print('Oops! Provide a valid input')
        is_continue = 'y'
