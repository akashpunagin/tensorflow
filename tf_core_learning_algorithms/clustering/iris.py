# Dataset
# This specific dataset seperates flowers into 3 different classes of species.
# - Setosa
# - Versicolor
# - Virginica
#
# The information about each flower is the following.
# - sepal length
# - sepal width
# - petal length
# - petal width

from __future__ import absolute_import, division, print_function, unicode_literals
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print('--------------------------------\n')

# constants to help later
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# # using keras to grab our datasets and read them into a pandas dataframe
# train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
# test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Load Dataset
train = pd.read_csv('iris_training.csv',names=CSV_COLUMN_NAMES , header=0)
test = pd.read_csv('iris_test.csv',names=CSV_COLUMN_NAMES , header=0)


# Labels
train_y = train.pop('Species')
test_y = test.pop('Species')

print('train df:\n', train.head())
print('describe train dataset :\n', train.describe())
print('train dataset shape : ', train.shape)
print('test dataset shape : ', test.shape)

# Training vs Testing Data

# Generate feature columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key, dtype=tf.float32))
print('\nfeature_columns:')
pprint(my_feature_columns)

# Creating input function, convert pd.DataFrame to tf.data.Dataset
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Building the Model, build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# Training, include a lambda to avoid creating an inner function in input function
print('\nTRAINING:\n')
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# Evaluation
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('Test set accuracy: {:0.3f}'.format(eval_result['accuracy']))
pprint(eval_result)

# Prediction for manual input
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("\Prediction for manual input.\nType numeric values for features as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit():
        valid = False
    predict[feature] = [float(val)]

manual_predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pred_dict in manual_predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))

# Predicting for test data
predictions = list(classifier.predict(input_fn=lambda: input_fn(test)))
probs = pd.Series([pred['probabilities'][1] for pred in predictions])
cls_ids = pd.Series([pred['class_ids'][0] for pred in predictions])

# Visualizing probabilities
plt.figure()
probs.plot(kind='hist', bins=20, title='predicted probabilities')
# plt.show()

plt.figure()
cls_ids.plot(kind='hist', bins=20, title='predicted class ids')
plt.show()
