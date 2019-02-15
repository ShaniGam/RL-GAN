import copy
import glob
import os
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from roadfighter_a2c.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from roadfighter_a2c.envs import make_env, get_img_counter, set_generator
from roadfighter_a2c.model import Policy
from roadfighter_a2c.storage import RolloutStorage

import roadfighter_a2c.algo as algo

from os import listdir
from os.path import isfile, join

args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def print_gan_log(j, final_rewards, gan_file):
    end = time.time()
    total_num_steps = (j + 1) * args.num_processes * args.num_steps
    print(
        "GAN iter {}, Updates {}, num timesteps {}, reward {:.1f}".
        format(gan_file, j, total_num_steps,
               final_rewards.max()))
    with open("roadfighter_a2c/log_{}.txt".format(args.gan_dir), 'a+') as f:
        f.write(
        "GAN iter {}, Updates {}, num timesteps {}, reward {:.1f}\n".
        format(gan_file, j, total_num_steps,
               final_rewards.max()))

def save_checkpoint(state, filename):
    torch.save(state, filename)

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    torch.set_num_threads(1)

    if args.test_gan:
        gan_path = args.gan_models_path + args.gan_dir + '/checkpoints'
        files = [join(gan_path, f).split('_')[1].split('.')[0] for f in listdir(gan_path) if
                 isfile(join(gan_path, f)) and f.startswith('gen')]
        gan_file = files.pop(0)

        envs = [make_env(i, args, True, gan_file)
                for i in range(args.num_processes)]
    else:
        envs = [make_env(i, args)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    agent = algo.A2C(actor_critic, args.value_loss_coef,
                     args.entropy_coef, lr=args.lr,
                     eps=args.eps, alpha=args.alpha,
                     max_grad_norm=args.max_grad_norm)

    if args.load:
        fname = 'roadfighter_a2c/' + args.env_name +  '1.pth.tar'
        print(fname)
        if os.path.isfile(fname):
            checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
            actor_critic.load_state_dict(checkpoint['state_dict'])
            for param in actor_critic.parameters():
                param.requires_grad = True
            print ("model loaded")

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
        current_obs[:,-shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    total_rewards = torch.zeros([args.num_processes, 1])
    reward = 0.0

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states, _ = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step],
                        deterministic=args.test_gan)
            cpu_actions = action.squeeze(1).cpu().numpy()
            total_rewards += reward

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)


            if args.test_gan:
                if done:
                    print_gan_log(j, final_rewards, gan_file)
                    gan_file = files.pop(0)
                    set_generator(gan_file)
                    j = 0

        if not args.test_gan: 
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.gamma)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.save and j % args.save_interval == 0 and args.save_dir != "":
            print ("Saving model")
            save_path = os.path.join(args.save_dir)
            save_checkpoint({
                'state_dict': actor_critic.state_dict(),
                #               'optimizer': optimizer.state_dict(),
            }, args.env_name + ".pth.tar")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if args.collect_images:
            if get_img_counter() > args.num_collected_imgs:
                break

        if not args.test_gan and j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss))
            if args.log:
                with open("log_lvl{}.txt".args.level, "a+") as f:
                    f.write("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}\n".
                        format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss))


if __name__ == "__main__":
    main()
