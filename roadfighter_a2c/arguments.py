import argparse

import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--num-frames', type=int, default=10e8,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='RoadFighterLvl',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default=None,
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--log', action='store_true', default=False,
                        help='log results')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load a trained model')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the model')
    parser.add_argument('--test-gan', action='store_true',
                        help='run a3c with gans.')
    parser.add_argument('--gan-dir', default='./',
                        help='directory of saved gan models')
    parser.add_argument('--gan-models-path', type=str, default='unit/outputs/',
                        help='gan models path')
    parser.add_argument('--gan-imitation-file', type=str, default='',
                        help='the gan file used for the imitation learning algo')
    parser.add_argument('--num-traj-steps', type=int, default=30000,
                        help='number of steps to remember when running trajectories')
    parser.add_argument('--traj-num', type=int, default=10,
                        help='number of trajectories to run')
    parser.add_argument('--db-batch-size', type=int, default=4,
                        help='batch size during supervised training')
    parser.add_argument('--off-policy-interval', type=int, default=100,
                        help='off policy interval, off policy updates every n on policy updates (default: 100)')
    parser.add_argument('--det-score', type=float, default=4000.,
                        help='minimum score to add trajectory to database')
    parser.add_argument('--super-test', type=int, default=20,
                        help='number of supervised training iterations before testing')
    parser.add_argument('--start-rl-thr', type=float, default=10000.,
                        help='minimum score to stop supervised training and start RL training')
    parser.add_argument('--max-super-iter', type=int, default=np.inf,
                        help='number of supervised training iterations before testing')
    parser.add_argument('--log-name', default='RoadFighterLvl2.log',
                        help='logger file name')
    parser.add_argument('--super-during-rl', action='store_true', default=False,
                        help='update rl network using supervised learning')
    parser.add_argument('--level', default='1',
                        help='road fighter level to choose')
    parser.add_argument('--traj-coef', type=float, default=0.75,
                        help='coefficient to collecting trajectory')
    parser.add_argument('--off-policy-coef', type=float, default=0.6,
                        help='coefficient for off-policy updates')
    parser.add_argument('--collect-images', action='store_true',
                        help='collect images for GAN training.')
    parser.add_argument('--num-collected-imgs', type=int, default=10,
                        help='number of images to collect (default: 100K)')
    args = parser.parse_args()

    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
