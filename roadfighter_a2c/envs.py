import gym
import numpy as np
from gym.spaces.box import Box

from baselines import bench
import retro
import cv2

import unit.transform_image as t
import os

obs = None
use_gan = False

model = t.TestModel()

img_counter = 1
args = None

def get_img_counter():
    return img_counter

def set_generator(which_epoch):
    """
    setting and loading pre-trained generator
    :param gan_dir: gan directory name
    :param which_epoch: the epoch the model was saved
    """
    print("setting generator")
    print(which_epoch)
    gan_path = args.gan_models_path + args.gan_dir
    model.initialize(gan_path, which_epoch)
    global use_gan
    use_gan = True


def set_gan_off():
    """
    stop using gan for translating images
    """
    global use_gan
    use_gan = False


def make_env(rank, arguments, use_gan=False, which_epoch=''):
    """
    creates environment
    :param rank: # of the current process
    :param use_gan: True if using gan for translation, False otherwise
    :param which_epoch: the epoch the model was saved
    :param arguments: arguments received from the user
    """
    def _thunk():
        global args
        args = arguments
        env = retro.make(game='RoadFighter-Nes', state='RoadFighter.Level{}'.format(args.level))

        env.seed(args.seed + rank)

        if args.log_dir is not None:
            env = bench.Monitor(env, os.path.join(args.log_dir, str(rank)), allow_early_resets=True)
        else:
            env = bench.Monitor(env, None, allow_early_resets=True)

        env = AtariRescale84x84(env)
        env = NormalizedEnv(env)

        if use_gan:
            set_generator(which_epoch)

        if args.collect_images:
            if not os.path.exists('unit/datasets/roadfighter-lvl' + args.level):
                os.makedirs('unit/datasets/roadfighter-lvl' + args.level)

        return env

    return _thunk


class Normalize():
    def __init__(self):
        super(Normalize, self).__init__()
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        observation = observation.astype(np.float32)
        observation *= (1.0 / 255.0)
        observation = np.moveaxis(observation, -1, 0)

        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
                          observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
                         observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

normalize = Normalize()

def _process_frame84(frame):
    """
    pre-processing of the frame
    :param frame: input image received from the game
    """
    global img_counter
    frame = frame[:, :184]
    frame = cv2.resize(frame, (84, 84))

    if args.collect_images and img_counter <= args.num_collected_imgs:
        np.save('unit/datasets/roadfighter-lvl' + args.level + '/img_' + str(img_counter), frame)
        with open('unit/datasets/roadfighter-lvl' + args.level + "/train_images.txt", 'a') as f:
            f.write('img_' + str(img_counter) + '.npy\n')
        img_counter += 1

    # translate image from target to source
    if use_gan:
        global obs
        obs = normalize._observation(frame.copy())
        frame = model.transform(frame)

    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale84x84, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [3, 84, 84])

    def _observation(self, observation):
        return _process_frame84(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0


    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

def _obs():
    """
    get original observations
    :return: normalized observation of the target tasks
    """
    return obs
