from typing import List

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

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    # TODO: create an actual sampling method
    return self.buffer[0]

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    # TODO: create an actual sampling method
    return -1