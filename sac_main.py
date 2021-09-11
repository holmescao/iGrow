import argparse
import gym
import os
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# from logger import Logger

import SAC.sac_module.utils as utils
from TenSim.simulator import PredictModel
from SAC.sac_module.replay_buffer import ReplayBuffer
from SAC.sac_module.sac import SACAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.set_num_threads(1)


# TesnorboardX
writer = SummaryWriter(
    logdir='SAC/sac_module/runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'WUR',
                                                     'SAC'))


def sim_env(version, base_tmp_folder):
    direcrory = base_tmp_folder+'/models/'
    model_dir = direcrory + version
    model_path = model_dir + '/model/'
    scaler_dir = model_dir + '/scaler/'

    ten_env = PredictModel(model1_dir=model_path+'simulator_greenhouse.pkl',
                           model2_dir=model_path+'simulator_crop_front.pkl',
                           model3_dir=model_path+'simulator_crop_back.pkl',
                           scaler1_x=scaler_dir+'greenhouse_x_scaler.pkl',
                           scaler1_y=scaler_dir+'greenhouse_y_scaler.pkl',
                           scaler2_x=scaler_dir+'crop_front_x_scaler.pkl',
                           scaler2_y=scaler_dir+'crop_front_y_scaler.pkl',
                           scaler3_x=scaler_dir+'crop_back_x_scaler.pkl',
                           scaler3_y=scaler_dir+'crop_back_y_scaler.pkl',
                           linreg_dir=model_path+'/PARsensor_regression_paramsters.pkl',
                           weather_dir=model_path+'/weather.npy')

    return ten_env


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        action = np.rint(action)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        action = action.astype(int)

        return action


def test(args):
    seed = 9
    utils.set_seed_everywhere(seed)
    env = sim_env(args.version, args.base_tmp_folder)
    env = NormalizedActions(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    action_range = [
        -1, 1
    ]
    agent = SACAgent(obs_dim=obs_dim, action_dim=act_dim,
                     action_range=action_range)
    agent.load_model("SAC/sac_model/sac_actor_%d_Exp" % args.seed,
                     "SAC/sac_model/sac_critic_%d_Exp" % args.seed)
    obs = env.reset()
    done = False
    episode_reward = 0
    rew_list = []
    while not done:
        with utils.eval_mode(agent):
            action = agent.act(obs, sample=False)
        obs, rew, done, _ = env.step(action)
        episode_reward += rew
        rew_list.append(episode_reward)
    plt.plot(rew_list)
    plt.show()


def main(args):
    utils.set_seed_everywhere(args.seed)

    env = sim_env(args.version, args.base_tmp_folder)
    env = NormalizedActions(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        -1, 1
    ]

    agent = SACAgent(obs_dim=obs_dim, action_dim=act_dim,
                     action_range=action_range)
    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape,
                                 50000,
                                 device=device)
    step = 0
    start_step = 2000
    eval_freq = 10
    max_test_episode = 0
    interval = 4
    for i in range(args.total_episode):
        episode_reward = 0
        obs = env.reset()
        done = False
        while not done:
            if step < start_step:
                temp_list = np.random.uniform(-0.1, 0.1, 24//interval)
                co2_list = np.random.uniform(-0.4, -0.2, 24//interval)
                illu_list = np.array([-1 for _ in range(24//interval)])
                irri_list = np.array([-1 for _ in range(24//interval)])
                action = np.hstack((temp_list, co2_list, illu_list, irri_list))
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            if len(replay_buffer) > start_step:
                agent.update(replay_buffer, writer, step)

            action = action.repeat(interval)
            next_obs, reward, done, _ = env.step(action)
            done_no_max = float(done)
            episode_reward += reward
            replay_buffer.add(obs, action, reward, next_obs, done,
                              done_no_max)
            obs = next_obs
            step += 1

        writer.add_scalar('train/episode_reward', episode_reward, step)
        print('step: {}, episode_reward: {}'.format(step, episode_reward))

        if i % eval_freq == 0:
            print('--------------------------------')
            print('start evaluation')
            print('--------------------------------')
            episode_reward_list = []
            evaluate_episodes = 3
            for episode in range(evaluate_episodes):
                obs = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    with utils.eval_mode(agent):
                        action = agent.act(obs, sample=True)
                    obs, rew, done, _ = env.step(action)
                    episode_reward += rew
                episode_reward_list.append(episode_reward)
                print(action)
            writer.add_scalar('eval_mean/episode',
                              np.mean(episode_reward_list), step)
            writer.add_scalar('eval_std/episode',
                              np.std(episode_reward_list), step)
            print('episode: {} evaluate value: {} evaluate std: {}'.format(i, np.mean(episode_reward_list),
                                                                           np.std(episode_reward_list)))
            if np.mean(episode_reward_list) > max_test_episode:
                max_test_episode = np.mean(episode_reward_list)
                agent.save_model(actor_path=args.save_dir+"sac_actor_%d_Exp" % args.seed,
                                 critic_path=args.save_dir+"sac_critic_%d_Exp" % args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--save_dir", default="SAC/sac_model/", type=str)
    parser.add_argument("--version", default="incremental", type=str)
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--total_episode", default=8000, type=int)
    args = parser.parse_args()

    main(args)
    # test(args)
