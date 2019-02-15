"""
Based on: https://github.com/mingyuliutw/UNIT Paper: https://arxiv.org/abs/1703.00848
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
from unit.utils import get_config
from torch.autograd import Variable
from unit.trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float()
    image_numpy = image_numpy.data.numpy()[0]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


class TestModel():
  def initialize(self, gan_dir, which_epoch):
    """
    initialize the model with the hyperparameters from the '.yaml' file
    :param gan_dir: the directory of the gan models
    :param which_epoch: the epoch the model was saved
    """
    print(gan_dir, which_epoch)
    config = get_config('unit/configs/' + gan_dir.split('/')[-1] + '.yaml')
    config['vgg_model_path'] = '.'
    self.trainer = UNIT_Trainer(config)
    # Prepare network
    state_dict = torch.load("{}/checkpoints/gen_{}.pt".format(gan_dir, which_epoch), map_location=lambda storage, loc: storage)
    self.trainer.gen_a.load_state_dict(state_dict['a'])
    self.trainer.gen_b.load_state_dict(state_dict['b'])
    if torch.cuda.is_available():
      self.trainer.cuda()
    self.trainer.gen_a.eval()
    self.trainer.gen_b.eval()

  def transform(self, input):
    """
    translate input image to output domain
    :param input: input image
    """
    input = np.float32(input)
    input = input.transpose((2, 0, 1))
    final_data = ((torch.FloatTensor(input)/255.0)-0.5)*2

    final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2)))
    if torch.cuda.is_available():
      final_data = final_data.cuda()

    content, _ = self.trainer.gen_a.encode(final_data)

    outputs = self.trainer.gen_b.decode(content)

    output_img = outputs[0].data.cpu().numpy()
    out_img = np.uint8(255 * (np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5))

    return out_img
