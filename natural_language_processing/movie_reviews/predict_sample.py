import math
import re
from nltk.corpus import stopwords
# from bs4 import BeautifulSoup
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the model
model = tf.keras.models.load_model('models/movie_reviews_without_mask.h5')

print('\n------------------------------------------------\n')

# Load dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

# The dataset info includes an encoder (bound method)
encoder = info.features['text'].encoder

# Helper functions to predict
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
    return predictions[0][0]

def make_prediction():
    default_text = "This is a good-natured, albeit unrealistic, family film that both kids and adults will enjoy if they're OK with the violence, profanity, and disrespectful behavior within the family."

    sample_pred_text = input('Enter a review... (Enter to select default review)\n')

    if(sample_pred_text == ""):
        sample_pred_text = default_text

    predictions = sample_predict(sample_pred_text, pad=True)

    print('\nREVIEW : ', sample_pred_text)
    if(predictions > 0):
        print("\nThis is a positive review by {:.2f} units".format(math.fabs(predictions)))
    else:
        print("\nThis is a neative review by {:.2f} units".format(math.fabs(predictions)))

make_prediction()

is_continue = 'y'
while(is_continue == 'y'):
    is_continue = input('\nDo you want to continue? [y/n] : ')
    if(is_continue == 'y'):
        make_prediction()
    elif(is_continue == 'n'):
        print("Seems like you've seen enough")
    else:
        print('Oops! Provide a valid input')
        is_continue = 'y'
