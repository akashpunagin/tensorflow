import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow as tf
keras = tf.keras

print('\n-------------------------------------------------------------------\n')

# Load dataset, 80% of data-raw_train, 10% of data-raw_validation, and 10% of data-raw_test
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

print('\nDataset loaded successfully...\n')

print('Before formatting data :')
print('raw_train : ',raw_train)
print('raw_validation : ',raw_validation)
print('raw_test : ',raw_test)

# int2str is a Bound methood
get_label_name = metadata.features['label'].int2str
print('\nBound method : ',get_label_name)

print('Label-0 : ',get_label_name(0))
print('Label-1 : ',get_label_name(1))

# Displaying first two images and labels from the training set
plt.figure()
for i, (image, label) in enumerate(raw_train.take(2)):
    plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.title(get_label_name(label))
# plt.show()

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

print('\nAfter formatting data:')
print('train : ',train)
print('validation : ',validation)
print('test : ',test)

print('\nComparing shape of raw imgaes and formatted images :')
for img, label in raw_train.take(2):
    print("Original shape:", img.shape)
for img, label in train.take(2):
    print("New shape:", img.shape)

# shuffle and batch the training data
# batch validation and test data
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Inspect a batch of data:
for image_batch, label_batch in train_batches.take(1):
    print('\nimage_batch.shape : ',image_batch.shape)


# Create the base model from the pre-trained convolutional neural network (convnets)
# MobileNet V2 model is trained on the ImageNet dataset consisting of 1.4M images and 1000 classes
# weights trained on ImageNet
# include_top=False argument, loading a network that doesn't include the classification layers at the top, which is ideal for feature extraction.
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features

feature_batch = base_model(image_batch)
print('image_batch.shape after applying feature extractor : ',feature_batch.shape)

# print('\nbase_model summary before freezing convolutional base')
# base_model.summary()

# Feature extraction
# freeze the convolutional base, it prevents the weights in a given layer from being updated during training
base_model.trainable = False

# print('\nbase_model summary after freezing convolutional base')
# base_model.summary()

# Add a classification head
# GlobalAveragePooling2D - Global average pooling operation for spatial data, converts the features to a single 1280-element vector per image.
# Returns 2D tensor with shape (batch_size, channels)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)
print('image_batch.shape after feature extractor and pooling : ',feature_batch_average.shape)

# Apply a Dense layer to convert the features into a single prediction per image.
# Activation function is not needed because prediction will be treated as a logit, positive numbers predict class 1, negative numbers predict class 0.
prediction_layer = tf.keras.layers.Dense(1)

prediction_batch = prediction_layer(feature_batch_average)
print('image_batch.shape after feature extractor, pooling and Dense layer : ',prediction_batch.shape)

# Stack the feature extractor, GlobalAveragePooling2D, and Dense layer
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# Compile the model
# Since there are two classes binary cross-entropy loss is used
# from_logits=True because the model provides a linear output
print('\nCompiling the model...')
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

n_trainable_weights = model.trainable_variables[0].numpy()
n_trainable_bias = model.trainable_variables[1].numpy()
print('\nNumber of trainable weights : ',len(n_trainable_weights))
print('Number of trainable bias : ', len(n_trainable_bias))

# Evaluate the model with validation data
initial_epochs = 10
validation_steps=20
print('\nEvaluating for validation data...')
loss0,accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print('\nFor validation data :')
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train the model
print('\nTraining the model...')
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

model.save("models/cats_vs_dogs_pretrained.h5")  # save the model
print('Model saved...')
# new_model = tf.keras.models.load_model('cats_vs_dogs_pretrained.h5')

# Learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle('Before Fine Tuning', fontsize=12)

axs[0].plot(acc, label='Training Accuracy')
axs[0].plot(val_acc, label='Validation Accuracy')
axs[0].legend(loc='lower right')
axs[0].set_ylabel('Accuracy')
axs[0].set_ylim([min(axs[0].set_ylim()),1])
axs[0].set_title('Training and Validation Accuracy')

axs[1].plot(loss, label='Training Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[0].legend(loc='upper right')
axs[1].set_ylabel('Cross Entropy')
axs[1].set_ylim([0,1.0])
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('epoch')
# plt.show()

# Observation -  validation metrics are clearly better than the training metrics because
# training metrics report the average for an epoch, while validation metrics are evaluated after the epoch
# so validation metrics see a model that has trained slightly longer.


# Fine Tuning - to increase performance train (or fine-tune) the weights of the top layers of the pre-trained model alongside the training of the classifier you added
# The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.
# NOTE: This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable
# If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly,
# the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.

# Un-freeze the top layers of the model
base_model.trainable = True

# Set the bottom layers to be un-trainable
# Number of layers in base_model
print("\nNumber of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

# Compile the model using a much lower learning rate. (necessary for the changes to take effect)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

print('\nAfter Fine Tuning ...')
model.summary()

# Continue training the model
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)

model.save("models/cats_vs_dogs_pretrained_with_fine_tuning.h5")  # save the model
print('Model saved...')
# new_model = tf.keras.models.load_model('cats_vs_dogs_pretrained_with_fine_tuning.h5')

# Learning curves after fine tuning
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle('After Fine Tuning', fontsize=12)

axs[0].plot(acc, label='Training Accuracy')
axs[0].plot(val_acc, label='Validation Accuracy')
axs[0].set_ylim([0.8, 1])
axs[0].plot([initial_epochs-1,initial_epochs-1],
          axs[0].set_ylim(), label='Start Fine Tuning')
axs[0].legend(loc='lower right')
axs[0].set_title('Training and Validation Accuracy')

axs[1].plot(loss, label='Training Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[1].set_ylim([0, 1.0])
axs[1].plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
axs[1].legend(loc='upper right')
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('epoch')
plt.show()
