import logging
import time

import ray
import torch
from torch import optim

from config import MuZeroConfig
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
  optimizer = optim.SGD(network.parameters(),
      lr=learning_rate, momentum=config.momentum)

  while ray.get(replay_buffer.get_buffer_size.remote()) == 0:
    logging.debug('Waiting on replay buffer to be filled...')
    time.sleep(30)

  ### Main training loop ###
  for step in range(config.training_steps):
    batch_future = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
    if step % config.checkpoint_interval == 0:
        storage.save_network.remote(step, network)
    batch = ray.get(batch_future)
    update_weights(optimizer, network, batch, config.weight_decay)


def update_weights(optimizer: optim.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, actions, targets in batch:
    print('OMG MADE IT THIS FAR', image.shape, actions.shape, type(targets), targets)
    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(
        image)
    predictions = [(1.0, value, reward, policy_logits)]

  #   # Recurrent steps, from action and previous hidden state.
  #   for action in actions:
  #     value, reward, policy_logits, hidden_state = network.recurrent_inference(
  #         hidden_state, action)
  #     predictions.append((1.0 / len(actions), value, reward, policy_logits))

  #     hidden_state = scale_gradient(hidden_state, 0.5)

  #   for prediction, target in zip(predictions, targets):
  #     gradient_scale, value, reward, policy_logits = prediction
  #     target_value, target_reward, target_policy = target

  #     l = (
  #         scalar_loss(value, target_value) +
  #         scalar_loss(reward, target_reward) +
  #         tf.nn.softmax_cross_entropy_with_logits(
  #             logits=policy_logits, labels=target_policy))

  #     loss += scale_gradient(l, gradient_scale)

  # for weights in network.get_weights():
  #   loss += weight_decay * tf.nn.l2_loss(weights)

  # optimizer.minimize(loss)

# def scale_gradient(tensor, scale):
#   """Scales the gradient for the backward pass."""
#   return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

# def scalar_loss(prediction, target) -> float:
#   # MSE in board games, cross entropy between categorical values in Atari.
#   return -1