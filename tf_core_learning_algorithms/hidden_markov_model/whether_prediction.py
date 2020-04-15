# The components of a markov model
# 1. States: In each markov model we have a finite set of states
# These states could be something like "warm" and "cold" or "high" and "low" or even "red", "green" and "blue"
# These states are "hidden" within the model, which means we do not direcly observe them.
#
# 2. Observations: Each state has a particular outcome or observation associated with it based on a probability distribution
# An example of this is the following: On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.
#
# 3. Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state
# An example is the following: a cold day has a 30% chance of being followed by a hot day and a 70% chance of being follwed by another cold day.

# To create a hidden markov model we need:
#     1. States
#     2. Transition Distribution
#     3. Observation Distribution

# This model tries to predict the temperature on each day
# In this model - Cold days are encoded by a 0 and hot days are encoded by a 1

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# The first day in our sequence has an 80% chance of being cold
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # [cold,hot]

# A cold day has a 30% chance of being followed by a hot day
# A hot day has a 20% chance of being followed by a cold day
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]]) # [cold-day[cold, hot], hot-day[cold, hot]]

# On each day the temperature is normally distributed with
# mean and standard deviation 0 and 5 on a cold day and
# mean and standard deviation 15 and 10 on a hot day.
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) # loc - mean, scale - standard devitation

# Creating the Hidden Markov Model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7) # The number of steps represents the number of days that we would like to predict information for

mean = model.mean()

print('\n-----------------------------------------------\n')
print('Predicted temperature of 7 days with given probabilities:')
with tf.compat.v1.Session() as sess:
  for i, temp in enumerate(mean.numpy()):
      print(f'Day-{i+1} : {temp}')
