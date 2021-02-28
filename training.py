import logging
import time

import numpy as np
import ray
import torch
from torch.nn import functional as F
from torch import optim

from config import MuZeroConfig
from env import Action
from models import Network
from replay_data import ReplayBuffer


@ray.remote
class SharedStorage(object):
  def __init__(self):
    self.max_count = 5
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      logging.warn('Attempted to retrieve the latest network, ' + \
          'but none have been saved yet!')
      return None

  def save_network(self, step: int, network: Network):
    # Delete any networks above `self.max_count`
    n_networks = len(self._networks.keys())
    if n_networks >= self.max_count:
      ordered_keys = list(sorted(self._networks.keys()))
      for i in range(n_networks + 1 - self.max_count):
        del self._networks[ordered_keys[i]]

    self._networks[step] = network

def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  ### Pre-training setup ###
  network = ray.get(storage.latest_network.remote())
  network.to('cuda') # TODO: Add a config setting for training device

  learning_rate = config.lr_init * config.lr_decay_rate**(
      network.training_steps() / config.lr_decay_steps)
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
      momentum=config.momentum, weight_decay=config.weight_decay)

  while ray.get(replay_buffer.get_buffer_size.remote()) == 0:
    logging.debug('Waiting on replay buffer to be filled...')
    time.sleep(2)#30)

  ### Main training loop ###
  for step in range(config.training_steps):
    batch_future = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
    if step % config.checkpoint_interval == 0:
        storage.save_network.remote(step, network)
    batch = ray.get(batch_future)
    logging.debug('updating weights')
    batch_update_weights(optimizer, network, batch)

def update_weights(optimizer: optim.Optimizer, network: Network, batch):
  optimizer.zero_grad()

  value_loss = 0
  reward_loss = 0
  policy_loss = 0
  for image, actions, targets in batch:
    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(image)
    predictions = [(1.0 / len(batch), value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
    for action in actions:
      value, reward, policy_logits, hidden_state = network.recurrent_inference(
          hidden_state, action)
      # TODO: Try not scaling this for efficiency
      # Scale so total recurrent inference updates have the same weight as the on initial inference update
      predictions.append((1.0 / len(actions), value, reward, policy_logits))

      hidden_state = scale_gradient(hidden_state, 0.5)

    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = \
          (torch.tensor(item, dtype=torch.float32, device=value.device.type) \
          for item in target)

      # Past end of the episode
      if len(target_policy) == 0:
        break

      value_loss += gradient_scale * scalar_loss(value, target_value)
      reward_loss += gradient_scale * scalar_loss(reward, target_reward)
      policy_loss += gradient_scale * cross_entropy_with_logits(policy_logits, target_policy)
      
      # print('val -------', value, target_value, scalar_loss(value, target_value))
      # print('rew -------', reward, target_reward, scalar_loss(reward, target_reward))
      # print('pol -------', policy_logits, target_policy, cross_entropy_with_logits(policy_logits, target_policy))

  value_loss /= len(batch)
  reward_loss /= len(batch)
  policy_loss /= len(batch)

  total_loss = value_loss + reward_loss + policy_loss
  scaled_loss = scale_gradient(total_loss, gradient_scale)

  logging.info('Training step {} losses'.format(network.training_steps()) + \
      ' | Total: {:.5f}'.format(total_loss) + \
      ' | Value: {:.5f}'.format(value_loss) + \
      ' | Reward: {:.5f}'.format(reward_loss) + \
      ' | Policy: {:.5f}'.format(policy_loss))

  scaled_loss.backward()
  optimizer.step()
  network.increment_step()

def batch_update_weights(optimizer: optim.Optimizer, network: Network, batch):
  optimizer.zero_grad()

  value_loss = 0
  reward_loss = 0
  policy_loss = 0

  # Format training data
  image_batch = np.array([item[0] for item in batch])
  action_batches = np.array([item[1] for item in batch])
  target_batches = np.array([item[2] for item in batch])
  action_batches = np.swapaxes(action_batches, 0, 1)
  target_batches = target_batches.transpose(1, 2, 0)

  # Run initial inference
  values, rewards, policy_logits, hidden_states = network.batch_initial_inference(image_batch)
  predictions = [(1, values, rewards, policy_logits)]

  # Run recurrent inferences
  for action_batch in action_batches:
    values, rewards, policy_logits, hidden_states = network.batch_recurrent_inference(
        hidden_states, action_batch)
    predictions.append((1.0 / len(action_batches), values, rewards, policy_logits))

    hidden_states = scale_gradient(hidden_states, 0.5)

  # Calculate losses
  for target_batch, prediction_batch in zip(target_batches, predictions):
    gradient_scale, values, rewards, policy_logits = prediction_batch
    target_values, target_rewards, target_policies = \
        (torch.tensor(list(item), dtype=torch.float32, device=values.device.type) \
        for item in target_batch)
    
    gradient_scale = torch.tensor(gradient_scale, dtype=torch.float32, device=values.device.type)
    value_loss += gradient_scale * scalar_loss(values, target_values)
    reward_loss += gradient_scale * scalar_loss(rewards, target_rewards)
    policy_loss += gradient_scale * cross_entropy_with_logits(policy_logits, target_policies, dim=1)

  value_loss = value_loss.mean() / len(batch)
  reward_loss = reward_loss.mean() / len(batch)
  policy_loss = policy_loss.mean() / len(batch)

  total_loss = value_loss + reward_loss + policy_loss
  logging.info('Training step {} losses'.format(network.training_steps()) + \
      ' | Total: {:.5f}'.format(total_loss) + \
      ' | Value: {:.5f}'.format(value_loss) + \
      ' | Reward: {:.5f}'.format(reward_loss) + \
      ' | Policy: {:.5f}'.format(policy_loss))

  # Update weights
  total_loss.backward()
  optimizer.step()
  network.increment_step()

def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + tensor.detach() * (1 - scale)

# TODO: Change this loss for atari once I implement supports
def scalar_loss(prediction, target) -> float:
  return torch.square(target - prediction)

# Should use dim=0 for single batch, dim=1 for multiple batches
def cross_entropy_with_logits(prediction, target, dim=0):
  return -torch.sum(target * F.log_softmax(prediction, dim=dim), dim=dim)

if __name__ == '__main__':
  print('Cross Entropy Test:', \
      cross_entropy_with_logits(torch.tensor([0.0088, 0.1576, -0.0345, -0.0805]), \
      torch.tensor([0.0000, 0.1429, 0.4286, 0.4286])))

  in_shape = (8*3, 96, 96)
  action_space_size = 4
  network = Network(in_shape, action_space_size, 'cuda')

  batch_size = 3
  rollout_len = 5

  batch = []
  for i in range(batch_size):
    img = np.ones(in_shape)
    actions = [Action(2) for _ in range(rollout_len)] 
    # (value, reward, empirical_policy)
    targets = [(0.7, 0.5, [0.25, 0.25, 0.25, 0.25]) for _ in range(rollout_len+1)]

    batch.append((img, actions, targets))
    
  optimizer = optim.SGD(network.parameters(), lr=0.001,
      momentum=0.9, weight_decay=1e-4)

  for i in range(1000):
    batch_update_weights(optimizer, network, batch)