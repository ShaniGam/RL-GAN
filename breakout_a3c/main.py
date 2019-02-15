from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import breakout_a3c.my_optim as my_optim
from breakout_a3c.envs import create_atari_env
from breakout_a3c.model import ActorCritic
from breakout_a3c.test import test
from breakout_a3c.train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4',
                    help='environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--save', action='store_true',
                    help='save the model.')
parser.add_argument('--test', action='store_true',
                    help='test the model.')
parser.add_argument('--variation', default='standard',
                    help='breakout variations: standard|constant-rectangle|moving-square|green-lines|diagonals')
parser.add_argument('--ft-setting', default='from-scratch',
                    help='fine-tuning setting: from-scratch|full-ft|random-output|partial-ft|partial-random-ft')
parser.add_argument('--collect-images', action='store_true',
                    help='collect images for GAN training.')
parser.add_argument('--num-collected-imgs', type=int, default=10,
                    help='number of images to collect (default: 100K)')
parser.add_argument('--test-gan', action='store_true',
                    help='run a3c with gans.')
parser.add_argument('--gan-dir', type=str, default='none',
                    help='gan dir name')
parser.add_argument('--gan-models-path', type=str, default='unit/outputs/',
                    help='gan dir name')

if __name__ == '__main__':
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
#    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name, args)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    # load a pre-trained model according to the ft-setting
    if args.ft_setting != 'from-scratch':
        fname = 'breakout_a3c/' + args.env_name + '.pth.tar'
        print(fname)
        if os.path.isfile(fname):
            checkpoint = torch.load(fname)
            shared_model.load_state_dict(checkpoint['state_dict'])
            for param in shared_model.parameters():
                param.requires_grad = True
            if 'partial' in args.ft_setting:
                for param in shared_model.conv1.parameters():
                    param.requires_grad = False
                for param in shared_model.conv2.parameters():
                    param.requires_grad = False
                for param in shared_model.conv3.parameters():
                    param.requires_grad = False

            if 'random' in args.ft_setting:
                shared_model.init_output_layers()

                if 'partial' in args.ft_setting:
                    shared_model.init_conv_lstm()

            print("model was loaded successfully")

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    if args.test:
        p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
        p.start()
        processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
