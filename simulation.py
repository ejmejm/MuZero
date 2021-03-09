import logging

import ray

from config import MuZeroConfig
from env import Game
from mcts import run_mcts, Node
from models import Network
from recording import TensorboardLogger
from replay_data import ReplayBuffer
from training import SharedStorage

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, writer: TensorboardLogger):
  logging.debug('Starting self play loop.')

  while True:
    # TODO: Stagger getting the newest network by n episodes for efficienty
    network = ray.get(storage.latest_network.remote())
    # TODO: Add an option to use GPUs for rollout actors
    network = network.to('cpu')

    logging.debug('Starting new game.')
    game = play_game(config, network)
    del game.environment # No reason to have this take up extra space
    writer.record_sim_results.remote(sum(game.rewards), len(game.history))
    logging.debug('Finished game, storing in replay buffer.')

    # TODO: Stagger saving to replay_buffer by n episodes for efficiency
    replay_buffer.save_game.remote(game)


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
                network.initial_inference(last_observation).numpy())
    root.add_exploration_noise(config)

    # logging.debug('Running MCTS on step {}.'.format(len(game.history)))
    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = root.select_action(config, len(game.history), network)
    game.apply(action)
    game.store_search_statistics(root)

  logging.info('Finished episode at step {} | cumulative reward: {}' \
      .format(len(game.obs_history), sum(game.rewards)))

  return game