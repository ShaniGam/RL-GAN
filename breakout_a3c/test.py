import time
from collections import deque

import torch
import torch.nn.functional as F

from breakout_a3c.envs import create_atari_env
from breakout_a3c.model import ActorCritic

from os import listdir
from os.path import isfile, join

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    if args.test_gan:
        log_name = 'breakout_a3c/' + args.gan_dir
        gan_path = args.gan_models_path + args.gan_dir + '/checkpoints'
        files = [join(gan_path, f).split('_')[1].split('.')[0] for f in listdir(gan_path) if
                 isfile(join(gan_path, f)) and f.startswith('gen')]
        gan_file = files.pop(0)
        env = create_atari_env(args.env_name, args, True, gan_file)
    else:
        env = create_atari_env(args.env_name, args)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            if args.test_gan:
                iterations = gan_file
                print("Model {}, Score {}\n".format(iterations, reward_sum))
                with open('breakout_a3c/' + log_name + '.txt', 'a') as f:
                    f.write("Model {}, Score {}\n".format(iterations, reward_sum))
            else:
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            if args.save:
                torch.save({
                    'state_dict': model.state_dict(),
                }, args.env_name + ".pth.tar")

            if args.test_gan:
                if files:
                    gan_file = files.pop(0)
                else:
                    break
                env = create_atari_env(args.env_name, args, True, gan_file)
            else:
                time.sleep(30)

        state = torch.from_numpy(state)
