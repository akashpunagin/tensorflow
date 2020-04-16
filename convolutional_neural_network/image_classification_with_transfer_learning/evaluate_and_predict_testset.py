import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow as tf
keras = tf.keras


model = tf.keras.models.load_model('models/cats_vs_dogs_pretrained_with_fine_tuning.h5')
print('\n-------------------------------------------------------------------\n')
print('\nModel loaded successfully...')
model.summary()

# Load dataset, 80% of data-raw_train, 10% of data-raw_validation, and 10% of data-raw_test
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Length of raw test dataset
n_raw_test_samples = [i for i,_ in enumerate(raw_test)][-1] + 1
print('\nLength of raw test dataset : ', n_raw_test_samples)

get_label_name = metadata.features['label'].int2str

# Format the Data
IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/(255/2)) - 1 # rescale the input channels to a range of [-1,1]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) # Resize the images to a fixed input size
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# batch validation and test data
BATCH_SIZE = 32
test_batches = test.batch(BATCH_SIZE)

# Evaluate the model with validation data
test_steps=20
print('\nEvaluating for test data...')
loss0,accuracy0 = model.evaluate(test_batches, steps=test_steps)

print('\nFor test data :')
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

class_names = ['cat', 'dog']

# Making Predictions for all test images
print('\nPredicting for test batches...')
predictions = model.predict(test_batches)

# negative value - cat, positive value - dog
get_class_names = lambda x : class_names[0] if x[0] < 0 else class_names[1]

predicted_class = list(map(get_class_names, predictions))

true_class = []
for img, label in test.take(n_raw_test_samples):
    true_class.append(get_label_name(label))

df_pred = pd.DataFrame({'True': list(true_class), 'Predicted': list(predicted_class)})
print('\nTrue VS Predicted DataFrame :\n', df_pred.head(10))


# Predicting for singe image
# COLOR = 'white'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label, num):
  prediction = model.predict(np.array([image]))
  predicted_class = get_class_names(prediction)
  print(f'\nImage-{num}:\nExcpected : {correct_label}\nPredicted : {predicted_class}')
  show_image(image, correct_label, predicted_class, num)

def show_image(img, label, pred, num):
  plt.figure()
  plt.imshow(img)
  plt.title(f'Image-{num}\nExcpected : {label}')
  plt.xlabel(f'Predicted : {pred}')
  plt.xticks([])
  plt.yticks([])
  plt.show()

def get_number():
  while True:
    num = input(f'Pick a number between 0 and {n_raw_test_samples}: ')
    if num.isdigit():
      num = int(num)
      if 0 <= num <= n_raw_test_samples:
        return int(num)
    else:
      print("Try again...")

print('\nPredicting for single image:')
num = get_number()
for img, label in test.take(num):
    pass

image = img
label = get_label_name(label)
predict(model, image, label, num)

is_continue = 'y'
while(is_continue == 'y'):
    is_continue = input('Do you want to continue? [y/n] : ')
    if(is_continue == 'y'):
        num = get_number()
        for img, label in test.take(num):
            pass
        image = img
        label = get_label_name(label)
        predict(model, image, label, num)
    elif(is_continue == 'n'):
        print("Seems like you've seen enough")
    else:
        print('Oops! Provide a valid input')
        is_continue = 'y'
