import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

# Helper function to plot graphs
def plot_graphs(history, metric):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    # plt.show()

# Load dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

print('\n------------------------------------------------\n')

train_examples, test_examples = dataset['train'], dataset['test']

# print(info)

# The dataset info includes an encoder (bound method)
encoder = info.features['text'].encoder
print('\ntype(encoder) : ',type(encoder))

# Vocabulary size
print(f'\nVocabulary size: {encoder.vocab_size}')

# Demonstration of encoder
sample_string = 'Hello TensorFlow.'
encoded_string = encoder.encode(sample_string)
decoded_string = encoder.decode(encoded_string)

print('\nSample string : ', sample_string)
print('Encoded string : ', encoded_string)
print('Decoded string : ', decoded_string)

print('\nIndex with its representation : ')
for index in encoded_string:
    print(f'{index} - {encoder.decode([index])}')

# Prepare the data for training, creating batches of the encoded strings
# padded_batch - zero-pad the sequences to the length of the longest string in the batch

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (train_examples.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None],[])))
test_dataset = (test_examples.padded_batch(BATCH_SIZE, padded_shapes=([None],[])))

# Creating the model
# embedding layer - stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors
# After training these vectors words with similar meanings will have similar vectors
# Bidirectional layer - propagates the input forward and backwards through the RNN layer and then concatenates the output
# LSTM (Long short-term memory network) - LSTM's were created as a method to reduce short-term memory problem
# RNN's are good for processing sequence data for predictions but suffers from short-term memory

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=64, mask_zero=True), #
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
print('\nCompiling the model...')
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# Train the model
print('\nTraining the model...')
history = model.fit(train_dataset, epochs=50,
                    validation_data=test_dataset,
                    validation_steps=30)

# Save the model
model.save('models/movie_reviews_with_mask.h5')
print('\nModel saved...')

# Evaluate the model in test set
test_loss, test_acc = model.evaluate(test_dataset)

print('\nTest Loss: {}'.format(test_loss)) # Test Loss: 0.39861352903687436
print('Test Accuracy: {}'.format(test_acc)) # Test Accuracy: 0.8610799908638


# Making predictions
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return predictions

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)


# plots
plot_graphs(history, 'accuracy')
plt.savefig('plots/accuracy_with_mask.jpg')
plot_graphs(history, 'loss')
plt.savefig('plots/loss_with_mask.jpg')
plt.show()
