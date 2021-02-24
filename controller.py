import logging
import time

import ray

from env import Game
from models import make_uniform_network, Network
from replay_data import ReplayBuffer
from config import make_gym_atari_config, MuZeroConfig
from mcts import Node, run_mcts

AGENT_LOG_PATH = 'logs/agent.log'
CONTROLLER_LOG_PATH = 'logs/controller.log'
TRAINER_LOG_PATH = 'logs/trainer.log'

@ray.remote
class SharedStorage(object):
  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network

##################################
####### Part 1: Self-Play ########

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  print('Starting self play loop')
  print('TYPES:', type(SharedStorage), type(replay_buffer))

  while True:
    # TODO: Stagger getting the newest network by n episodes for efficienty
    network = ray.get(storage.latest_network.remote())

    print('NETWORK TYPE:', type(network))
    logging.debug('Starting new game.')
    game = play_game(config, network)
    del game.environment # No reason to have this take up extra space
    logging.debug('Finished game, storing in replay buffer.')

    # TODO: Stagger saving to replay_buffer by n episodes for efficiency
    game_ref = ray.put(game)
    replay_buffer.save_game.remote(game_ref)
    
# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = Game.from_config(config)

  while not game.terminal() and len(game.history) < config.max_moves:
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    last_observation = game.make_image(-1)
    root.expand(game.to_play(), game.legal_actions(),
                network.initial_inference(last_observation))
    root.add_exploration_noise(config)

    logging.debug('Running MCTS on step {}.'.format(len(game.history)))
    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = root.select_action(config, len(game.history), network)
    game.apply(action)
    game.store_search_statistics(root)
  return game

# TODO: fill out function
@ray.remote #(num_cpus=2, num_gpus=0.2)
def launch_simulation_process(config, shared_storage, replay_buffer):
  logging.basicConfig(filename=AGENT_LOG_PATH, level=logging.DEBUG)
  run_selfplay(config, shared_storage, replay_buffer)

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
  shared_storage = SharedStorage.remote()
  initial_network = Network(config.obs_shape, config.action_space_size)
  shared_storage.save_network.remote(0, ray.put(initial_network))
  replay_buffer = ReplayBuffer.remote(config)

  # TODO: Try making the config a remote then passing it through
  # TODO: Try making the config a remote then pass it in through a ray.put(...)

  sim_processes = []
  for _ in range(1): #config.num_actors):
    print('launching sim')
    proc = launch_simulation_process.remote(config, shared_storage, replay_buffer)
    sim_processes.append(proc)
  # time.sleep(5)
  # ray.wait(sim_processes)

  while True:
    buffer_size = ray.get(replay_buffer.get_buffer_size.remote())
    print('Buffer size:', buffer_size)
    time.sleep(20)

  # train_network(config, storage, replay_buffer)

  # return storage.latest_network()

if __name__ == '__main__':
  logging.basicConfig(filename=CONTROLLER_LOG_PATH, level=logging.DEBUG)

  ray.init()

  config = make_gym_atari_config('Breakout-v0')
  muzero(config)
