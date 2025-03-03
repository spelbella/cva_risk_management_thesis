import collections
import gymnasium as gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, models
from typing import Any, List, Sequence, Tuple

import pickle

from env import tradingEng

from path_datatype import Path

""" Environment Setup"""
# Define environment
with open("1.6kRunDemo.pkl","rb") as fp:
    paths = pickle.load(fp)
env = tradingEng(paths)

# Now setup how we read/write our action and observation space from/to dict
def read_obs():
    obs = env._get_obs()
    s_pos = obs['Swaption Position']
    q_pos = obs['Q Position'] 
    s_val = obs['Swaption Value']
    q_val = obs['Q Value']
    r = obs['r']

    obs = s_pos + q_pos + s_val + q_val + [r]
    return tf.reshape(tf.convert_to_tensor(obs))
n_obs = len(read_obs())

def format_action(action):
    s_pos = action[0:9]
    q_pos = action[9:]
    action = {"Swaption Position" : s_pos,  "Q Position" : q_pos}
    return action
n_actions = 18

""" Returns"""
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards,dtype=np.float32)
    running_return = 0
    for t in reversed (range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

"""NN setup"""
# Define learning parameters for NN
gamma = 0.99
learning_rate = 0.01
num_episodes = 1000
batch_size = 32

inputs = keras.Input(shape = (n_obs,))
dense = layers.Dense(2*n_obs, activation = 'relu')
x = dense(inputs)
x = layers.Dense(2*n_obs, activation="relu")(x)
x = layers.Dense(2*n_obs, activation="relu")(x)
x = layers.Dense(2*n_obs, activation="relu")(x)
outputs = layers.Dense(18)(x)

model = keras.Model(inputs = inputs, outputs=outputs, name = 'Test Model')
model.summary()
model.compile(loss='mse', optimizer=keras.optimizers.Adam())
optimizer = tf.keras.optimizers.Adam(learning_rate)

""" Training the NN """
def run(self):
    for path in range(0,1000):
        env.reset()
        state = read_obs()
        
        reward = []
        done = False

        while not done:
            action = format_action(model(state))
            _, loc_reward, done = env.step()
            reward.append(reward)
            state = read_obs()
        
        returns = compute_returns(reward,gamma)
        grads = tf.GradientTape(returns, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(returns[0])


run(None)
