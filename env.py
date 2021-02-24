from typing import List

import cv2
import gym
import numpy as np

# from config import MuZeroConfig
from config import ATARI_DEFAULT_CROP_SIZE, \
    ATARI_DEFAULT_FRAMESTACK_SIZE, MuZeroConfig
from mcts import Node

class Action(object):
  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

def generate_action_range(action_space_size):
  return [Action(i) for i in range(action_space_size)]

# TODO: Implement Player class for 2 player games
class Player(object):
  pass

# Implement Environment class
class Environment(object):
  def __init__(self, env_name):
    self.env = gym.make(env_name)
    self.curr_obs = self.env.reset()
    self.done = False

  def step(self, action: Action):
    self.curr_obs, reward, self.done, _ = self.env.step(action.index)
    return reward

  def reset(self):
    self.curr_obs = self.env.reset()

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    # TODO: Store and return correct player for multi-player games
    return Player()
  
# TODO: Consider preprocessing in base env to save time on recalculation
def atari_preprocess_obs(state_index: int, obs_history: List,
                         crop_size=ATARI_DEFAULT_CROP_SIZE,
                         n_framestack=ATARI_DEFAULT_FRAMESTACK_SIZE):
    if state_index < 0:
      state_index = len(obs_history) - state_index

    start_idx = max(0, state_index + 1 - n_framestack)
    end_idx = state_index + 1
    target_frames = obs_history[start_idx:end_idx]

    # Resize frames
    for i, frame in enumerate(target_frames):
      target_frames[i] = cv2.resize(frame, crop_size, interpolation=cv2.INTER_LINEAR)
    target_frames = np.array(target_frames, dtype=np.float32)

    # Color scaling
    target_frames = target_frames / 255.0
    # Rearange color channels to front of height x width
    target_frames = target_frames.transpose(0, 3, 1, 2)
    # If necessary, add in any extra start frames
    under_clip = n_framestack - len(target_frames)
    if under_clip > 0:
      extra_frames = []
      for _ in range(under_clip):
        extra_frames.append(target_frames[0:1])
      target_frames = np.concatenate(extra_frames + [target_frames], axis=0)
    # Squash first 2 dimensions
    dims = target_frames.shape
    target_frames = target_frames.reshape(dims[0] * dims[1], dims[2], dims[3])

    return target_frames

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, config: MuZeroConfig):
    self.env_name = config.env_name
    self.environment = Environment(self.env_name)  # Game specific environment.
    self.obs_history = [self.environment.curr_obs] # Observations at each step
    self.history = [] # Action taken at each step
    self.rewards = [] # Reward taken at each step
    self.child_visits = [] # Each element is a list of fractions of the time each action was taken for a given step
    self.root_values = [] # Each element is the average value of a simulated state for the given step
    self.action_space_size = config.action_space_size
    self.action_space = generate_action_range(self.action_space_size)
    self.discount = config.discount

    if config.obs_preprocessing_type and config.obs_preprocessing_type.lower() == 'atari':
      self.make_image = lambda state_index: atari_preprocess_obs(
          state_index, self.obs_history, config.obs_shape[1:])
    else:
      raise NotImplementedError('"atari" is currently the only supported preprocessing type!')

  def from_config(config: MuZeroConfig):
    return Game(config)

  def terminal(self) -> bool:
    return self.environment.done

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return self.action_space

  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    self.obs_history.append(self.environment.curr_obs)

  def store_search_statistics(self, root: Node):
    children_nodes = root.children.values()
    sum_visits = sum(child.visit_count for child in children_nodes) # Total playthroughs extending from root
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    raise NotImplementedError()

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)