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
from MarketGeneratingFunctions.path_datatype import Path

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
    q_pos = loc_action[9:18]
    ret_action = {"Swaption Position" : s_pos,  "Q Position" : q_pos}
    return ret_action
n_actions = 18

"""NN setup"""
class actorCritic(tf.keras.Model):
    """ Basic Gradient Network """
    
    def __init__(self):
        self.n_obs = n_obs
        self.n_outputs = n_actions

        super().__init__()

        self.common = layers.Dense(2*n_obs, activation="relu")
        self.pre_action = layers.Dense(n_obs*2, activation="relu")
        self.almost_action = layers.Dense(n_actions, activation="relu")
        self.more = layers.Dense(n_actions, activation="relu")
        self.action = layers.Softmax()
        self.pre_critic = layers.Dense(n_actions, activation="relu")
        self.pre_critic2p = layers.Dense(4, activation="relu")
        self.pre_critic2 = layers.Dense(4, activation="relu")
        self.pre_critic3 = layers.Dense(2, activation="sigmoid")
        self.critic = layers.Dense(1, activation='linear')

    def call(self, inputs:tf.Tensor) -> tuple[tf.Tensor]:
        common = self.common(inputs)
        pre_action = self.pre_action(inputs)
        almost_action = self.almost_action(pre_action)
        more = self.more(almost_action)
        action = self.action(more)

        pre_critic = self.pre_critic(common)
        pre_critic2p = self.pre_critic2p(pre_critic)
        pre_critic2 = self.pre_critic2(pre_critic2p)
        pre_critic3 = self.pre_critic3(pre_critic2)
        critic = self.critic(pre_critic3)
        return action, critic

model = actorCritic()

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
    critics = tf.TensorArray(dtype=np.float32, size=0, dynamic_size=True)

    state = tf.expand_dims(read_obs(), 0)
    for t in tf.range(10000):
        # Convert state into a batched tensor
        state = tf.expand_dims(read_obs(), 0)

        # Run the model
        action, critic_val = model(state)

        # Store the action and critic
        actions = actions.write(t, action)
        critics = critics.write(t, critic_val)
        
        # Apply the action
        state, reward, done = env_step(action)

        # Store reward
        rewards = rewards.write(t, reward)

        # Break at end of epoch
        if done:
            break

    actions = actions.stack()
    rewards = rewards.stack()
    critics = critics.stack()
    return actions, rewards, critics

def get_expected_return(rewards: tf.Tensor, gamma: float) -> tf.Tensor:
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype = tf.float32, size = n)

    rewards = tf.cast(rewards[::-1], dtype = tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward*(1/1e-6) + gamma * discounted_sum 
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i,discounted_sum)
    returns = returns.stack()[::-1]
    return returns

huber_loss = tf.keras.losses.Huber(reduction = tf.keras.losses.Reduction.SUM)
@tf.function
def compute_loss(actions, returns, critics):
    actor_loss = tf.math.reduce_mean(returns - critics)
    critic_loss = huber_loss(returns, critics)
    return actor_loss, critic_loss

""" Training """
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.005)

# Train and learn
min_episodes_criterion = 100
max_episodes = 100000
gamma = 0.99

@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:

    with tf.GradientTape() as tape:
        actions, rewards, critics = run_episode(model)
        returns = get_expected_return(rewards, gamma)
        
        # Format fix
        actions, rewards, critics = [tf.expand_dims(x,1) for x in [actions, rewards, critics]]

        # Calculate loss
        act_loss, crit_loss = compute_loss(actions, returns, critics)
        loss_value = act_loss + crit_loss

    # Compute gradients
    grads = tape.gradient(loss_value, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_mean(rewards)

    return episode_reward

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    _, info = env.reset()
    initial_state = read_obs()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = train_step(
        initial_state, model, optimizer, gamma, None)

    episodes_reward.append(episode_reward)
    running_reward = np.mean(np.array(episodes_reward))

    t.set_postfix(episode_reward=episode_reward)

    # Show the average episode reward every 10 episodes
    if i % 10 == 0:
      print(f'Episode {i}: reward: {running_reward}')