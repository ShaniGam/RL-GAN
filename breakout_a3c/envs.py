import cv2
import gym
import numpy as np
from gym.spaces.box import Box
import os

change_counter = 1
positions = [20,40,60]
current_position = 0

img_counter = 1
args = None

gan_translate = False
model = None

def get_img_counter():
    return img_counter


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id, arguments, test_gan=False, gan_file=''):
    global args, gan_translate, model
    args = arguments
    if args.collect_images:
        if not os.path.exists('unit/datasets/breakout-' + args.variation):
            os.makedirs('unit/datasets/breakout-' + args.variation)
            os.makedirs('unit/datasets/breakout-' + args.variation + '/trainA')
            os.makedirs('unit/datasets/breakout-' + args.variation + '/trainB')

    if test_gan:
        import unit.transform_image as t
        model = t.TestModel()
        gan_path = args.gan_models_path + args.gan_dir
        model.initialize(gan_path, gan_file)
        gan_translate = True

    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    return env


def _process_frame42(frame):
    global img_counter
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))

    if args.collect_images and img_counter <= args.num_collected_imgs:
        np.save('unit/datasets/breakout-' + args.variation + '/trainB/img_' + str(img_counter), frame)

    if args.variation == 'constant-rectangle':
        for i in range(2):
            for j in range(4):
                frame[i + 60][j + 60][0] = 0
                frame[i + 60][j + 60][1] = 191
                frame[i + 60][j + 60][2] = 255

    elif args.variation == 'moving-square':
        global change_counter
        global current_position
        for i in range(3):
            for j in range(3):
                frame[60 + i][positions[current_position] + j][0] = 255
                frame[60 + i][positions[current_position] + j][1] = 0
                frame[60 + i][positions[current_position] + j][2] = 0
        if (change_counter % 1000) == 0:
            change_counter = 1
            current_position += 1
            if current_position > 2:
                current_position = 0
        change_counter += 1

    elif args.variation == 'green-lines':
        for i in range(30, 77):
            if i % 8 == 0:
                j = i
                l = 4
                while j < 77 and l < 76:
                    frame[i][l][0] = 102
                    frame[i][l][1] = 203
                    frame[i][l][2] = 50
                    j += 1
                    l += 1

    elif args.variation == 'diagonals':
        for i in range(30, 77):
            if i % 8 == 0:
                j = i
                l = 4
                while j < 77 and l < 76:
                    frame[j][l][0] = 210
                    frame[j][l][1] = 203
                    frame[j][l][2] = 50
                    j += 1
                    l += 1

    if args.collect_images and img_counter <= args.num_collected_imgs:
        np.save('unit/datasets/breakout-' + args.variation + '/trainA/img_' + str(img_counter), frame)
        with open('unit/datasets/breakout-' + args.variation + "/train_images.txt", 'a') as f:
            f.write('img_' + str(img_counter) + '.npy\n')
        img_counter += 1

    if gan_translate:
        frame = model.transform(frame)

    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [3, 80, 80])

    def _observation(self, observation):
        return _process_frame42(observation)
