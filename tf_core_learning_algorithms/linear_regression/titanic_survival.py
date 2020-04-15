from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
from pprint import pprint

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

print('--------------------------------\n')

# Load dataset.
dftrain = pd.read_csv('train.csv') # training data
dfeval = pd.read_csv('eval.csv') # testing data

# Labels
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print('dftrain :\n', dftrain.head())
print('describe dftrain :\n', dftrain.describe())
print('dftrain shape : ', dftrain.shape)
print('dfeval shape : ', dfeval.shape)

# Exploratroy Data Analysis
plt.figure()
dftrain.age.hist(bins=20)
# plt.show()
# Observation - Most passengers are in their 20's or 30's

plt.figure()
dftrain.sex.value_counts().plot(kind='barh')
# plt.show()
# Observation - Most passengers are male

plt.figure()
dftrain['class'].value_counts().plot(kind='barh')
# plt.show()
# Observation - Most passengers are in "Third" class

plt.figure()
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()
# Observation - Females have a much higher chance of survival


# Training vs Testing Data
# Generate feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(key=feature_name, dtype=tf.float32))
print('\nfeature_columns:')
pprint(feature_columns)

# The Training Process
# Creating input function, convert pd.DataFrame to tf.data.Dataset
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the Model - create a linear estimtor by passing the feature columns we created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the Model
print('\nTRAINING:\n')
linear_est.train(train_input_fn)  # train

# Evaluating the model
# keras.evaluate() is for evaluating your trained model. Its output is accuracy or loss, not prediction to your input data.
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data
print('Test set accuracy : ', result['accuracy'])  # the result variable is simply a dict of stats about our model
pprint(result)

# Predicting
# keras.predict() actually predicts, and its output is target value, predicted from your input data.
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
cls_ids = pd.Series([pred['class_ids'][0] for pred in pred_dicts])

# Visualizing probabilities
plt.figure()
probs.plot(kind='hist', bins=20, title='predicted probabilities')
# plt.show()

plt.figure()
cls_ids.plot(kind='hist', bins=20, title='predicted class ids')
plt.show()
