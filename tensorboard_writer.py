from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
  def __init__(self, log_dir):
    self.writer = SummaryWriter(log_dir=log_dir)

  def write_loss(self, iteration, group_name, name, losses=None):
    tag = '{}/{}'.format(group_name, name)
    scalars_dict = {}
    
    if losses is not None:
      for k, l in losses.items():
        if l is not None:
          scalars_dict[k] = l

    self.writer.add_scalars(tag, scalars_dict, iteration)

  def write_audio(self, iteration, group_name, name, snd_tensor, sample_rate):
    tag = '{}/{}'.format(group_name, name)
    self.writer.add_audio(tag, snd_tensor, global_step = iteration, sample_rate=sample_rate)
  
  def close(self):
    self.writer.close()
