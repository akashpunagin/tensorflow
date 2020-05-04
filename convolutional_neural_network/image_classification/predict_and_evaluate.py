import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

model = tf.keras.models.load_model('models/cifar_classification.h5')

print('\n-------------------------------------------------------------------\n')

# Load Dataset, split into traing set and testing set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Label values
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Evaluate model
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
