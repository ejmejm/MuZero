import gym
from gym.spaces.discrete import Discrete
import ray
from typing import Optional, Tuple

from mcts import KnownBounds

ATARI_DEFAULT_CROP_SIZE = (96, 96)
ATARI_DEFAULT_FRAMESTACK_SIZE = 8

class MuZeroConfig(object):
  def __init__(self,
         action_space_size: int,
         max_moves: int,
         discount: float,
         dirichlet_alpha: float,
         num_simulations: int,
         batch_size: int,
         td_steps: int,
         num_actors: int,
         lr_init: float,
         lr_decay_steps: float,
         visit_softmax_temperature_fn,
         env_name: str,
         action_repeat: int = 1,
         training_steps: int = int(1e6),
         obs_shape: Tuple[int] = (8*3, 96, 96),
         obs_preprocessing_type: Optional[str] = None,
         known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.env_name = env_name
    self.action_space_size = action_space_size
    # Valid types: [None, 'atari']
    self.obs_preprocessing_type = obs_preprocessing_type
    self.obs_shape = obs_shape
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount
    self.action_repeat = action_repeat

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = training_steps
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

# def make_board_game_config(action_space_size: int, max_moves: int,
#                dirichlet_alpha: float,
#                lr_init: float) -> MuZeroConfig:

#   def visit_softmax_temperature(num_moves, training_steps):
#     if num_moves < 30:
#       return 1.0
#     else:
#       return 0.0  # Play according to the max.

#   return MuZeroConfig(
#     action_space_size=action_space_size,
#     max_moves=max_moves,
#     discount=1.0,
#     dirichlet_alpha=dirichlet_alpha,
#     num_simulations=800,
#     batch_size=2048,
#     td_steps=max_moves,  # Always use Monte Carlo return.
#     num_actors=3000,
#     lr_init=lr_init,
#     lr_decay_steps=400e3,
#     visit_softmax_temperature_fn=visit_softmax_temperature,
#     known_bounds=KnownBounds(-1, 1))


# def make_go_config() -> MuZeroConfig:
#   return make_board_game_config(
#     action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


# def make_chess_config() -> MuZeroConfig:
#   return make_board_game_config(
#     action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


# def make_shogi_config() -> MuZeroConfig:
#   return make_board_game_config(
#     action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)

# def make_atari_config() -> MuZeroConfig:

#   def visit_softmax_temperature(num_moves, training_steps):
#     if training_steps < 500e3:
#       return 1.0
#     elif training_steps < 750e3:
#       return 0.5
#     else:
#       return 0.25

#   return MuZeroConfig(
#     action_space_size=18,
#     max_moves=27000,  # Half an hour at action repeat 4.
#     discount=0.997,
#     dirichlet_alpha=0.25,
#     batch_size=1024,
#     td_steps=10,
#     num_actors=350,
#     lr_init=0.05,
#     lr_decay_steps=350e3,
#     visit_softmax_temperature_fn=visit_softmax_temperature)

def make_gym_atari_config(env_name: str) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 50e3:
      return 1.0
    elif training_steps < 75e3:
      return 0.5
    else:
      return 0.25

  env = gym.make(env_name)
  if not isinstance(env.action_space, Discrete):
      raise Exception('Environment must have a discrete action space!')

  return MuZeroConfig(
    action_space_size=env.action_space.n,
    max_moves=1000, # Could increase this
    discount=0.997,
    dirichlet_alpha=0.25,
    num_simulations=6,
    batch_size=32,
    td_steps=10,
    num_actors=8,
    lr_init=0.005,
    lr_decay_steps=int(35e3),
    training_steps=int(1e5),
    env_name=env_name,
    action_repeat=3,
    obs_preprocessing_type='atari',
    obs_shape=(ATARI_DEFAULT_FRAMESTACK_SIZE * 3,) + ATARI_DEFAULT_CROP_SIZE,
    visit_softmax_temperature_fn=visit_softmax_temperature)