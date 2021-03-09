import ray
from torch.utils.tensorboard import SummaryWriter

@ray.remote
class TensorboardLogger():
  def __init__(self):
    self.writer = SummaryWriter()
    self.sim_step = 0
    self.training_step = 0

  def record_sim_results(self, reward, n_steps):
    self.writer.add_scalar('Simulation / Reward', reward, self.sim_step)
    self.writer.add_scalar('Simulation / Episode Length', n_steps, self.sim_step)
    self.sim_step += 1

  def record_training_results(self, total_loss, value_loss, reward_loss, policy_loss):
    self.writer.add_scalar('Training / Total Loss', total_loss, self.training_step)
    self.writer.add_scalar('Training / Value Loss', value_loss, self.training_step)
    self.writer.add_scalar('Training / Reward Loss', reward_loss, self.training_step)
    self.writer.add_scalar('Training / Policy Loss', policy_loss, self.training_step)
    self.training_step += 1


