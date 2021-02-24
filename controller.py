import logging
import time

import ray

from env import Game
from models import Network
from replay_data import ReplayBuffer
from config import make_gym_atari_config, MuZeroConfig
from mcts import Node, run_mcts
from simulation import run_selfplay
from training import SharedStorage, train_network

AGENT_LOG_PATH = 'logs/agent.log'
CONTROLLER_LOG_PATH = 'logs/controller.log'
TRAINER_LOG_PATH = 'logs/trainer.log'


@ray.remote #(num_cpus=1, num_gpus=0)
def launch_actor_process(config, shared_storage, replay_buffer):
  logging.basicConfig(filename=AGENT_LOG_PATH, level=logging.DEBUG)
  run_selfplay(config, shared_storage, replay_buffer)

@ray.remote #(num_cpus=2, num_gpus=0.8)
def launch_trainer_process(config, shared_storage, replay_buffer):
  logging.basicConfig(filename=TRAINER_LOG_PATH, level=logging.DEBUG)
  train_network(config, shared_storage, replay_buffer)

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
  # Create core objects
  shared_storage = SharedStorage.remote()
  # TODO: Decide whether to use CPU or GPU for actor networks
  initial_network = Network(config.obs_shape, config.action_space_size, device='cpu')
  shared_storage.save_network.remote(0, initial_network)
  replay_buffer = ReplayBuffer.remote(config)

  # Spin up actor processes
  sim_processes = []
  for i in range(1): #config.num_actors):
    logging.debug('Launching actor #{}'.format(i+1))
    proc = launch_actor_process.remote(config, shared_storage, replay_buffer)
    sim_processes.append(proc)

  launch_trainer_process.remote(config, shared_storage, replay_buffer)

  # Update buffer size
  while True:
    buffer_size = ray.get(replay_buffer.get_buffer_size.remote())
    logging.debug('Buffer size: {}'.format(buffer_size))
    time.sleep(10)

  # return storage.latest_network()


if __name__ == '__main__':
  logging.basicConfig(filename=CONTROLLER_LOG_PATH, level=logging.DEBUG)

  ray.init()

  config = make_gym_atari_config('Breakout-v0')
  muzero(config)
