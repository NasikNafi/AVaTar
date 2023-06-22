import os
import torch
import numpy as np
from collections import deque

from test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

from src import algo, utils
from src.arguments import parser
from src.model import PPOnet
from src.storage import RolloutStorage
from src.envs import VecPyTorchProcgen
from src import agent_utils

def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    log_file = '-{}-{}-s{}'.format(args.env_name, args.algo, args.seed)
    logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)
    print("\nLog File: ", log_file)

    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
        num_levels=args.num_levels, start_level=args.start_level, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    obs_shape = envs.observation_space.shape
    if args.algo == 'avatar':
        actor_critic = PPOnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})
    
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    if args.algo == 'avatar':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    nsteps = torch.zeros(args.num_processes)
    for j in range(num_updates):
        actor_critic.train()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.algo == 'avatar':
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
           
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1
            nsteps[done == True] = 0
            rollouts.insert(obs, action, action_log_prob, value, \
                            reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        # generating gamma using a truncated normal distribution
        gammas = agent_utils.generate_gammas(args.mean_gamma, args.std_gamma, args.lower_gamma, args.upper_gamma, args.num_of_gamma)
        rollouts.compute_returns(next_value, gammas, args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Save Model
        if j == num_updates - 1 and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(envs, 'ob_rms', None)
            ], os.path.join(args.save_dir, "agent{}.pt".format(log_file)))

        # Save Logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("\nUpdate {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"\
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards)))

            # Log training stats
            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            # Log eval stats (on the full distribution of levels)
            eval_episode_rewards = evaluate(args, actor_critic, device)
            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            logger.dumpkvs()


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
