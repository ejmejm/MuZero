from typing import List

import numpy as np
import ray

from config import MuZeroConfig
from env import Action, Environment, Player, Game

@ray.remote
class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def get_buffer_size(self):
    return len(self.buffer)

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

  # TODO: Make a prioritized sampling method
  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    if len(self.buffer) == 0:
      return None

    sample_idx = np.random.randint(0, len(self.buffer))
    return self.buffer[sample_idx]

  # TODO: Make a prioritized sampling method
  def sample_position(self, game: Game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return np.random.randint(0, len(game.obs_history))