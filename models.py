import copy

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple

from env import Action, generate_action_range

class NetworkOutput(NamedTuple):
  value: torch.Tensor
  reward: torch.Tensor
  policy_logits: torch.Tensor
  hidden_state: torch.Tensor

  def numpy(self):
    output_reward = self.reward.cpu().detach().numpy()
    output_policy = self.policy_logits.cpu().detach().numpy()
    output_value = self.value.cpu().detach().numpy()

    return NetworkOutput(output_value, output_reward, output_policy, self.hidden_state)

class Network(object):
  def __init__(self,
               in_shape,
               action_space_size,
               device=None,
               models=None):
    self.in_shape = in_shape
    self.action_space_size = action_space_size
    self.action_space = generate_action_range(self.action_space_size)
    self.hidden_state_shape = (128, 6, 6)
    self.step = 0
    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
      self.device = device

    if models:
      self.representation_model = models[0]
      self.prediction_model = models[1]
      self.dynamics_model = models[2]
    else:
      self.representation_model = RepresentationModel(in_shape).to(self.device)
      self.prediction_model = PredictionModel(action_space_size, self.hidden_state_shape).to(self.device)
      self.dynamics_model = DynamicsModel(action_space_size, (128+1, 6, 6)).to(self.device)

  def initial_inference(self, image) -> NetworkOutput:
    assert image.shape == self.in_shape, 'Initial inference input shape should be {}, not {}!' \
        .format(self.in_shape, image.shape)

    formatted_obs = torch.tensor(image, device=self.device).unsqueeze(0).float()
    hidden_state = self.representation_model(formatted_obs)
    policy, value = self.prediction_model(hidden_state)

    output_hidden_state = hidden_state
    output_reward = torch.tensor(0, dtype=torch.float32, device=self.device)
    output_policy = policy[0]
    output_value = value[0][0]

    # representation + prediction function
    return NetworkOutput(output_value, output_reward, output_policy, output_hidden_state)

  def batch_initial_inference(self, image) -> NetworkOutput:
    assert image.shape[1:] == self.in_shape, 'Initial inference input shape should be {}, not {}!' \
        .format(self.in_shape, image.shape)

    formatted_obs = torch.tensor(image, device=self.device).float()
    hidden_states = self.representation_model(formatted_obs)
    policies, values = self.prediction_model(hidden_states)

    batch_size = hidden_states.shape[0]
    output_hidden_states = hidden_states
    output_rewards = torch.tensor([0] * batch_size, dtype=torch.float32, device=self.device)
    output_policies = policies
    output_values = values.squeeze()

    # representation + prediction function
    return NetworkOutput(output_values, output_rewards, output_policies, output_hidden_states)

  def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
    assert hidden_state.shape == (1,) + self.hidden_state_shape, \
        'Recurrent inference input shape should be {}, not {}!' \
        .format(self.hidden_state_shape, hidden_state.shape)

    if self.device != hidden_state.device.type:
      hidden_state = hidden_state.to(self.device)

    # Add encoded action to hidden state
    # TODO: Encode the actions for the past n frames
    # Update on this, shouldn't actually be necessary because the
    # encoded action is being used to generate the next hidden state

    # TODO: Improve the action encoding space efficiency
    encoded_action = torch.zeros((1, 1) + self.hidden_state_shape[1:],
        device=self.device, dtype=torch.float32)
    action_scale = action.index / (self.action_space_size - 1)
    encoded_action += action_scale

    encoded_state_action = torch.cat([hidden_state, encoded_action], dim=1)

    next_hidden_state, reward = self.dynamics_model(encoded_state_action)
    policy, value = self.prediction_model(next_hidden_state)

    output_hidden_state = next_hidden_state
    output_reward = reward[0][0]
    output_policy = policy[0]
    output_value = value[0][0]

    # representation + prediction function
    return NetworkOutput(output_value, output_reward, output_policy, output_hidden_state)

  def batch_recurrent_inference(self, hidden_states, actions: Action) -> NetworkOutput:
    assert hidden_states.shape[1:] == self.hidden_state_shape, \
        'Recurrent inference input shape should be {}, not {}!' \
        .format(self.hidden_state_shape, hidden_states[1:].shape)
    batch_size = hidden_states.shape[0]

    if self.device != hidden_states.device.type:
      hidden_states = hidden_states.to(self.device)

    # Add encoded action to hidden state
    # TODO: Encode the actions for the past n frames
    # Update on this, shouldn't actually be necessary because the
    # encoded action is being used to generate the next hidden state

    # TODO: Improve the action encoding space efficiency
    encoded_actions = torch.ones((batch_size,) + self.hidden_state_shape[1:],
        device=self.device, dtype=torch.float32)
    action_scales = [action.index / (self.action_space_size - 1) for action in actions]
    shape_padding = [1] * (len(encoded_actions.shape) - 1)
    action_scales = torch.tensor(action_scales,
        device=self.device, dtype=torch.float32) \
        .view([len(action_scales)] + shape_padding)
    encoded_actions *= action_scales
    encoded_actions = encoded_actions.unsqueeze(1)
  
    encoded_state_actions = torch.cat([hidden_states, encoded_actions], dim=1)

    next_hidden_states, rewards = self.dynamics_model(encoded_state_actions)
    policies, values = self.prediction_model(next_hidden_states)

    output_hidden_states = next_hidden_states
    output_rewards = rewards.squeeze()
    output_policies = policies
    output_values = values.squeeze()

    # representation + prediction function
    return NetworkOutput(output_values, output_rewards, output_policies, output_hidden_states)

  def increment_step(self):
    self.step += 1

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return self.step

  def to(self, device, inplace=False):
    if self.device == device:
      return self

    if inplace:
      self.representation_model.to(device)
      self.prediction_model.to(device)
      self.dynamics_model.to(device)
      self.device = device
      return self

    new_network = copy.deepcopy(self)
    return new_network.to(device, inplace=True)

  def parameters(self):
    return list(self.representation_model.parameters()) + \
           list(self.prediction_model.parameters()) + \
           list(self.dynamics_model.parameters())

# TODO: Implement networks for non-atari environments using a modifiable config

class ResidualBlock(nn.Module):
  """ Custom Linear layer but mimics a standard linear layer """
  def __init__(self, n_filters, kernel_size=3):
    super().__init__()

    if kernel_size % 2 == 0:
        raise ValueError('Residual blocks must use and odd kernel size!')

    self.n_filters = n_filters
    padding = int(kernel_size // 2)

    self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size, 1, padding)
    self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, 1, padding)

    self.batch_norm1 = nn.BatchNorm2d(n_filters)
    self.batch_norm2 = nn.BatchNorm2d(n_filters)

  def forward(self, x):
    # Convolution 1
    z = self.batch_norm1(x)
    z = F.relu(z)
    z = self.conv1(z)

    # Convolution 2
    z = self.batch_norm2(z)
    z = F.relu(z)
    z = self.conv2(z)
    z += x

    return z

# Source: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
# Use this over normal list or layers don't get registered
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class RepresentationModel(nn.Module):
  def __init__(self, in_shape=(8*3, 96, 96)):
    super().__init__()
    self.in_shape = in_shape
    self.n_residual_blocks = 16

    if in_shape[0] % 3 != 0 or in_shape[1] != 96 or in_shape[2] != 96:
      raise Exception('RepresentationModel currently only accepts inputs of shape (3n, 96, 96)')

    self.conv1 = nn.Conv2d(self.in_shape[0], 128, 3, 2, 1)
    self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
    self.residual_blocks = (ResidualBlock(128, 3) for _ in range(self.n_residual_blocks))
    self.residual_blocks = ListModule(*self.residual_blocks)
    self.pool_layers = [nn.AvgPool2d(2) for _ in range(2)]
    self.pool_layers = ListModule(*self.pool_layers)

    # for i, res_block in enumerate(self.residual_blocks):
    #   self.register_buffer('residual_block_{}'.format(i), res_block)

  def forward(self, x):
    z = self.conv1(x)
    z = F.relu(z)

    for i in range(0, 2):
      z = self.residual_blocks[i](z)

    z = self.conv2(z)
    z = F.relu(z)

    for i in range(2, 5):
      z = self.residual_blocks[i](z)

    z = self.pool_layers[0](z)

    for i in range(5, 8):
      z = self.residual_blocks[i](z)

    z = self.pool_layers[1](z)

    for i in range(8, self.n_residual_blocks):
      z = self.residual_blocks[i](z)

    return z

class DynamicsModel(nn.Module):
  def __init__(self, action_space_size=4, in_shape=(128+1, 6, 6)):
    super().__init__()
    self.in_shape = in_shape
    self.n_residual_blocks = 16

    if in_shape[0] <= action_space_size:
      raise ValueError('DynamicsModel input shape first dimmension must be at least ' + \
          '{} to account for action encoding'.format(action_space_size + 1))

    if in_shape[1] != 6 or in_shape[2] != 6:
      raise ValueError('DynamicsModel currently only accepts inputs of shape (n, 6, 6)')

    self.conv_layer = nn.Conv2d(in_shape[0], 128, 3, 1, 1)
    self.residual_blocks = [ResidualBlock(128, 3) for _ in range(self.n_residual_blocks)]
    self.residual_blocks = ListModule(*self.residual_blocks)
    self.hidden_state_conv = nn.Conv2d(128, 128, 3, 1, 1)

    self.reward_conv = nn.Conv2d(128, 256, 6, 1, 0)
    self.reward_linear = nn.Linear(256, 1)

  def forward(self, x):
    z = self.conv_layer(x)
    for res_block in self.residual_blocks:
      z = res_block(z)
    z = F.relu(z)

    # TODO: try using ReLU on hidden state output for Atari
    hidden_state = self.hidden_state_conv(z)
    reward_pred_z = self.reward_conv(z)
    reward_pred_z = F.relu(reward_pred_z)
    reward_pred_z = reward_pred_z.view(-1, 256)
    reward_pred = self.reward_linear(reward_pred_z)

    return hidden_state, reward_pred

class PredictionModel(nn.Module):
  def __init__(self, action_space_size=4, in_shape=(128, 6, 6)):
    super().__init__()
    self.in_shape = in_shape

    if in_shape[1] != 6 or in_shape[2] != 6:
      raise ValueError('DynamicsModel currently only accepts inputs of shape (n, 6, 6)')

    self.conv1 = nn.Conv2d(self.in_shape[0], 64, 3, 1, 1)
    self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)

    self.conv_output_size = 32 * self.in_shape[1] * self.in_shape[2]

    self.policy_layer = nn.Linear(self.conv_output_size, action_space_size)
    self.value_layer = nn.Linear(self.conv_output_size, 1)

  def forward(self, x):
    z = self.conv1(x)
    z = F.relu(z)
    z = self.conv2(z)
    z = F.relu(z)
    z = z.view(-1, self.conv_output_size)

    value_pred = self.value_layer(z)
    policy_pred = self.policy_layer(z)
    # policy_pred = F.softmax(policy_pred)

    return policy_pred, value_pred

if __name__ == '__main__':
  import numpy as np

  n_samples = 3
  in_shape = (8*3, 96, 96)
  action_space_size = 4

  rep = RepresentationModel(in_shape)
  pred = PredictionModel(action_space_size, (128, 6, 6))
  dynamics = DynamicsModel(action_space_size, (128+action_space_size, 6, 6))
  
  formatted_obs = torch.ones((n_samples,) + in_shape).float()

  hidden_state = rep(formatted_obs)
  print('hidden states shape:', hidden_state.shape)

  policy, value = pred(hidden_state)
  print('policies:', policy)
  print('values:', value)

  encoded_acts = torch.zeros((n_samples, action_space_size, 6, 6))
  encoded_state_acts = torch.cat([hidden_state, encoded_acts], dim=1)
  print('encoded state and acts shape:', encoded_state_acts.shape)

  next_hidden_state, reward = dynamics(encoded_state_acts)
  print('next hidden states shape:', next_hidden_state.shape)
  print('rewards:', reward)