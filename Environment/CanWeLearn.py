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
    return tf.convert_to_tensor(obs)
n_obs = len(read_obs())

def format_action(action):
    loc_action = np.squeeze(action)
    s_pos = loc_action[0:9]
    q_pos = loc_action[9:]
    ret_action = {"Swaption Position" : s_pos,  "Q Position" : q_pos}
    return ret_action
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
class proxGrad(tf.keras.Model):
    """ Basic Gradient Network """
    
    def __init__(self):
        self.n_obs = n_obs
        self.n_outputs = n_actions

        super().__init__()

        self.inputl = layers.Dense(self.n_obs, activation="relu")
        self.hiddenl = layers.Dense(2*self.n_obs)
        self.hiddenl2 = layers.Dense(self.n_obs)
        self.outputl = layers.Dense(self.n_outputs)

    def call(self, inputs:tf.Tensor) -> tuple[tf.Tensor]:
        x = self.inputl(inputs)
        x = self.hiddenl(x)
        x = self.hiddenl2(x)
        x = self.outputl(x)
        return x

model = proxGrad()

@tf.numpy_function(Tout=[tf.float32, tf.float32, tf.bool])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  _, reward, done, truncated, info = env.step(format_action(action))
  state = read_obs()
  state = np.array(state, np.float32)
  reward = np.array(reward, np.float32)
  done = done
  return (state, reward, done)

def run_episode(model):
    actions = tf.TensorArray(dtype=np.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=np.float32, size=0, dynamic_size=True)

    state = tf.expand_dims(read_obs(), 0)
    for t in tf.range(1000):
        # Convert state into a batched tensor
        state = tf.expand_dims(read_obs(), 0)

        # Run the model
        action = model(state)

        # Store the action
        actions = actions.write(t, action)
    
        # Apply the action
        state, reward, done = env_step(action)

        # Store reward
        rewards = rewards.write(t, reward)

        # Break at end of epoch
        if done:
            break

    actions = actions.stack()
    rewards = rewards.stack()
    return actions, rewards

def compute_loss(actions, rewards):
    return -tf.math.reduce_mean(rewards)

""" Training """
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.05)
    
@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:

    with tf.GradientTape() as tape:
        actions, rewards = run_episode(model)
        

        # Format fix
        actions, rewards = [tf.expand_dims(x,1) for x in [actions, rewards]]

        # Calculate loss
        loss = compute_loss(actions, rewards)

    # Compute gradients
    grads = tape.gradient(loss, model.trainable_variables)
    print(grads)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward

# Train and learn
min_episodes_criterion = 100
max_episodes = 1000
gamma = 0.99

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    _, info = env.reset()
    initial_state = read_obs()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = train_step(
        initial_state, model, optimizer, None, None)

    episodes_reward.append(episode_reward)
    running_reward = np.mean(np.array(episodes_reward))

    t.set_postfix(episode_reward=episode_reward)

    # Show the average episode reward every 10 episodes
    if i % 10 == 0:
      print(f'Episode {i}: reward: {running_reward}')