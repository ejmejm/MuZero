# def train_network(config: MuZeroConfig, storage: SharedStorage,
#                   replay_buffer: ReplayBuffer):
#   network = Network()
#   learning_rate = config.lr_init * config.lr_decay_rate**(
#       tf.train.get_global_step() / config.lr_decay_steps)
#   optimizer = tf.train.MomentumOptimizer(learning_rate, config.momentum)

#   for i in range(config.training_steps):
#     if i % config.checkpoint_interval == 0:
#       storage.save_network(i, network)
#     batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
#     update_weights(optimizer, network, batch, config.weight_decay)
#   storage.save_network(config.training_steps, network)


# def scale_gradient(tensor, scale):
#   """Scales the gradient for the backward pass."""
#   return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

# def scalar_loss(prediction, target) -> float:
#   # MSE in board games, cross entropy between categorical values in Atari.
#   return -1