'''
Author: your name
Date: 2021-06-25 15:39:10
LastEditTime: 2021-06-27 14:57:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NMI/vs_methods.py
'''
import gym
import argparse
import scipy.io as scio
import os
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from TenSim.utils.data_reader import TomatoDataset
from TenSim.simulator_gpu import PredictModel
from utils.common import mkdir, save_curve
from utils.plt_params import plt_fig_params, set_day_xtick
from GA.config import setting
from SAC.sac_module.sac import SACAgent
from SAC.sac_module import utils

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'


def env(version, base_tmp_folder):
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

    # def record(self):

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


def get_stems(files_path):
    with open(files_path, "r") as f:
        data = f.readlines()
    for line in data:
        if "stems" in line:
            stems = np.load(line.strip("\n"))
            break
    return stems


def Figure4(args):
    print("=============Figure4===============")
    save_dir = args.base_tmp_folder+'/figure4/'
    mkdir(save_dir)

    stems = get_stems(args.wur_champion_files)

    # curve
    figure4a_EGA_balance = get_EGA(args, stems)
    figure4a_SAC_balance = get_SAC(args, stems)
    figure4a_Automatoes_balance, _ = get_Automatoes(args, stems)
    figure4b_EGA_control = get_control(args, method='EGA', var='temp')
    figure4c_EGA_control = get_control(args, method='EGA', var='lmp')
    figure4b_Automatoes_control = get_control(
        args, method='Automatoes', var='temp')
    figure4c_Automatoes_control = get_control(
        args, method='Automatoes', var='lmp')

    plantcost = 4.29
    # save
    figure4a_EGA = {"xlabel": 'date', "ylabel": "euro/m2",
                    "x": range(len(figure4a_EGA_balance)),
                    "y": np.cumsum(figure4a_EGA_balance)-plantcost}
    figure4a_SAC = {"xlabel": 'date', "ylabel": "euro/m2",
                    "x": range(len(figure4a_SAC_balance)),
                    "y": np.cumsum(figure4a_SAC_balance)-plantcost}
    figure4a_Automatoes = {"xlabel": 'date', "ylabel": "euro/m2",
                           "x": range(len(figure4a_Automatoes_balance)),
                           "y": np.cumsum(figure4a_Automatoes_balance)-plantcost}
    figure4b_EGA = {"xlabel": 'hour', "ylabel": "oC",
                    "x": np.arange(len(figure4b_EGA_control)),
                    "y": figure4b_EGA_control}
    figure4c_EGA = {"xlabel": 'hour', "ylabel": "on=1,off=0",
                    "x": np.arange(len(figure4c_EGA_control)),
                    "y": figure4c_EGA_control}
    figure4b_Automatoes = {"xlabel": 'hour', "ylabel": "oC",
                           "x": np.arange(len(figure4b_Automatoes_control)),
                           "y": figure4b_Automatoes_control}
    figure4c_Automatoes = {"xlabel": 'hour', "ylabel": "on=1,off=0",
                           "x": np.arange(len(figure4c_Automatoes_control)),
                           "y": figure4c_Automatoes_control}

    save_curve_dir = save_dir + '/curve/'
    mkdir(save_curve_dir)
    save_curve(figure4a_EGA, save_curve_dir+'figure4a_EGA.pkl')
    save_curve(figure4a_SAC, save_curve_dir+'figure4a_SAC.pkl')
    save_curve(figure4a_Automatoes, save_curve_dir+'figure4a_Automatoes.pkl')

    save_curve(figure4b_EGA, save_curve_dir+'figure4b_EGA.pkl')
    save_curve(figure4c_EGA, save_curve_dir+'figure4c_EGA.pkl')
    save_curve(figure4b_Automatoes, save_curve_dir+'figure4b_Automatoes.pkl')
    save_curve(figure4c_Automatoes, save_curve_dir+'figure4c_Automatoes.pkl')

    # show
    sim_res = {"EGA": list(figure4a_EGA['y']),
               "SAC": list(figure4a_SAC['y']),
               "Automatoes": list(figure4a_Automatoes['y'])}

    control_res = {"EGA": [figure4b_EGA_control, figure4c_EGA_control],
                   "team": [figure4b_Automatoes_control, figure4c_Automatoes_control]}

    compare_plot(sim_res=sim_res,
                 control_res=control_res,
                 xticks_list=[list(range(0, 20000, 10000))+[20000],
                              list(range(0, 1000, 350))+[1000]],
                 save_fig_dir=save_dir)


def get_control(args, method, var):
    if method == 'EGA':
        ga_policy_name = 'global_seed@%d_NIND@%d_MAXGEN@%d_XOVR@%.1f_LINKDAY@%d' \
            % (args.seed, args.NIND, args.MAXGEN, args.XOVR, args.LINKDAY)
        X_best_sofar_path = os.path.join(
            args.GA_train_result_path, ga_policy_name)
        X = list(scio.loadmat(X_best_sofar_path)['policy'][0])
        ga_policy = np.array(X)
        policy = ga_policy.reshape((-1, 96))
    elif method == 'Automatoes':
        tmp_folder = os.path.join(
            args.base_tmp_folder, 'models/%s' % args.version)
        wur_tomato_reader = TomatoDataset(args.wur_team_files, tmp_folder)
        train_data = wur_tomato_reader.read_data(args.wur_team_files)
        full_train_x, _ = wur_tomato_reader.data_process(train_data)
        X = full_train_x[-1, :, 6:10]
        policy = X[:24*160, :]

    # 选择第80天开始的120小时的策略
    link_day = 5
    d = 80
    day_dim = 24
    link = []
    for c in range(3):
        if method == 'EGA':
            link_policy = policy[d:d+link_day, :]
            ga_col_policy = link_policy[:, c*day_dim:(c+1)*day_dim]
            link.append(ga_col_policy.flatten())
        elif method == 'Automatoes':
            link_policy = policy[d*24:(d+link_day)*24, :]
            team_col_policy = link_policy[:, c]
            link.append(team_col_policy)

    if var == 'temp':
        setpoints = link[0]
    elif var == 'lmp':
        setpoints = link[2]

    return setpoints


def get_EGA(args, stems):
    balance = ga_sim(args, stems)

    return balance


def get_SAC(args, stems):
    balance, _ = sac_sim(args, stems)

    return balance


def sac_sim(args, stems):
    ten_env = env(args.version, args.base_tmp_folder)
    obs = ten_env.reset(stems)
    ten_env = NormalizedActions(ten_env)

    obs_dim = ten_env.observation_space.shape[0]
    act_dim = ten_env.action_space.shape[0]
    action_range = [-1, 1]

    agent = SACAgent(obs_dim=obs_dim, action_dim=act_dim,
                     action_range=action_range)
    actor_path = os.path.join(args.sac_model_dir, args.sac_actor)
    critic_path = os.path.join(args.sac_model_dir, args.sac_critic)
    agent.load_model(actor_path, critic_path)

    done = False
    reward = []
    economic = {'balance': [],
                'gains': [],
                'variableCosts': [],
                'elecCost': [],
                'co2Cost': [],
                'heatCost': [],
                'laborCost': []}

    while not done:
        with utils.eval_mode(agent):
            action = agent.act(obs, sample=False)
        obs, r, done, ec = ten_env.step(action)
        reward.append(float(r))

        for k, v in ec.items():
            economic[k].append(v)

    return reward, economic


def ga_sim(args, stems):
    ga_policy_name = 'global_seed@%d_NIND@%d_MAXGEN@%d_XOVR@%.1f_LINKDAY@%d' \
        % (args.seed, args.NIND, args.MAXGEN, args.XOVR, args.LINKDAY)
    X_best_sofar_path = os.path.join(
        args.GA_train_result_path, ga_policy_name)
    X = list(scio.loadmat(X_best_sofar_path)['policy'][0])

    dims = 96
    policy = np.zeros((args.DAY_IN_LIFE_CYCLE*24, 4))
    for d in range(args.DAY_IN_LIFE_CYCLE):
        dayX = X[d*dims: (d+1)*dims]
        for varIdx in range(4):
            policy[d*24:(d+1)*24, varIdx] = dayX[varIdx*24:(varIdx+1)*24]

    ten_env = env(args.version, args.base_tmp_folder)
    balance, _ = simOurModel(policy, ten_env, stems)

    return balance


def simOurModel(period_action, ten_env, CropParams):
    ten_env.reset(CropParams)

    dims = 24
    day = 0
    done = False
    reward = []
    economic = {'balance': [],
                'gains': [],
                'variableCosts': [],
                'elecCost': [],
                'co2Cost': [],
                'heatCost': [],
                'laborCost': []}

    # 仿真整个周期，并获取最终reward
    while not done:
        a = period_action[day*dims: (day+1)*dims, :]  # 获取1天的策略
        a = a.reshape((-1), order='F')
        _, r, done, ec = ten_env.step(a)
        day += 1
        reward.append(float(r))

        for k, v in ec.items():
            economic[k].append(v)

    return reward, economic


def ga_params(parser):
    parser.add_argument(
        '--seed', default=setting['params_info']['seed'], help='random seed')
    parser.add_argument(
        '--NIND', default=setting['params_info']['NIND'], help='population size')
    parser.add_argument(
        '--MAXGEN', default=setting['params_info']['MAXGEN'], help='maximum generation')
    parser.add_argument(
        '--LINKDAY', default=setting['params_info']['LINKDAY'], help='copy days')
    parser.add_argument(
        '--XOVR', default=setting['params_info']['XOVR'], help='crossover probability')

    return parser


def sac_params(parser):
    parser.add_argument('--sac_actor', default="sac_actor_7900")
    parser.add_argument('--sac_critic', default="sac_critic_7900")

    return parser


def compare_plot(sim_res, control_res, xticks_list, save_fig_dir):
    # fig, axes
    mpl.rcParams.update(plt_fig_params)

    fig = plt.figure(figsize=(15, 8))
    layout = (2, 7)
    plt.subplot2grid(layout, (0, 0), rowspan=2, colspan=3)
    for r in range(layout[0]):
        plt.subplot2grid(layout, (r, 3), rowspan=1, colspan=2)

    # 更新参数
    props1 = {0: {"xlabel": "date",
                  "ylabel": "euro / m$^2$",
                  }
              }
    props2 = {
        0: {"xlabel": "hour",
            "ylabel": "$^\circ$C",
            },
        1: {"xlabel": "hour",
            "ylabel": "on=1, off=0",
            },
    }
    # colors = ['#ec4646', cm.viridis(0.3), '#eaac7f']
    colors = ['#ec4646', cm.viridis(0.3), cm.viridis(0.6)]
    #              curve_name[1]: cm.viridis(0.6),
    #              curve_name[2]: cm.cool(0.3),
    #              curve_name[3]: cm.winter(0.3)}

    alpha = 1
    plt_fig_style1 = {
        'Automatoes': dict(linestyle='--', lw=2.8, alpha=alpha, color=colors[0], label='Automatoes'),
        'EGA': dict(linestyle='-', lw=2.8, alpha=alpha, color=colors[1], label='EGA'),
        'SAC': dict(linestyle='-', lw=2.8, alpha=alpha, color=colors[2], label='SAC'), }
    plt_fig_style2 = {
        'Automatoes': dict(linestyle='--', lw=1.5, alpha=alpha, color=colors[0], label='Automatoes'),
        'EGA': dict(linestyle='-', lw=1.5, alpha=alpha, color=colors[1], label='EGA'),
        'SAC': dict(linestyle='-', lw=1.5, alpha=alpha, color=colors[2], label='SAC'), }

    names = ['(a) NetProfit', '(b) Temperature', '(c) Illumination']
    xticks = list(range(0, 120, 30)) + [120]
    yticks_list = [list(range(12, 33, 6)),
                   [0, 1], ]
    # draw基础类对象
    ga = sim_res['EGA']
    sac = sim_res['SAC']
    best_team = sim_res['Automatoes']
    for idx, ax in enumerate(fig.axes):
        if idx == 0:
            ax.plot(list(ga), **plt_fig_style1['EGA'])
            ax.plot(sac, **plt_fig_style1['SAC'])
            ax.plot(best_team, **plt_fig_style1['Automatoes'])

            # 美化
            ticks, labels = set_day_xtick(num=4,
                                          var_list=list(ga[:]),
                                          startDate='2019-12-16',
                                          endDate='2020-05-29')
            ax.set_xticks(ticks=ticks)
            ax.set_xticklabels(labels=labels)
            ax.set_yticks(ticks=list(range(-20, 25, 10)))
            ax.set_title(names[idx], y=-0.24, fontsize=25)

            min_xlim, max_xlim = ax.get_xlim()
            min_ylim, max_ylim = ax.get_ylim()
            xlim_length = abs(max_xlim - min_xlim)
            ylim_length = abs(max_ylim - min_ylim)
            aspect = xlim_length / ylim_length
            ax.set_aspect(aspect*0.9)

            ax.set(**props1[idx])  # 参数设置a
            ax.grid(linestyle="--", alpha=0.4)

        else:
            idx_ = idx-1
            ax.plot(control_res['EGA'][idx_], **plt_fig_style2['EGA'])
            ax.plot(control_res['team'][idx_], **plt_fig_style2['Automatoes'])
            # 美化
            ax.set_xticks(ticks=xticks)
            ax.set_yticks(ticks=yticks_list[idx_])
            ax.set_title(names[idx], y=-0.47, fontsize=25)
            ax.set(**props2[idx_])  # 参数设置

            min_xlim, max_xlim = ax.get_xlim()
            min_ylim, max_ylim = ax.get_ylim()
            xlim_length = abs(max_xlim - min_xlim)
            ylim_length = abs(max_ylim - min_ylim)
            aspect = xlim_length / ylim_length
            ax.set_aspect(aspect)
            ax.tick_params(axis='x', labelsize=22)
            ax.tick_params(axis='y', labelsize=22)
            ax.xaxis.label.set_size(25)
            ax.yaxis.label.set_size(25)
            ax.set_aspect(aspect*0.8)

        # for tick in ax.get_xticklabels():
        #     tick.set_rotation(15)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.xaxis.label.set_size(25)
        ax.yaxis.label.set_size(25)

    plt.tight_layout()

    # legend
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(0.1, -0.25), loc='upper left',
              ncol=3, framealpha=0, fancybox=False, fontsize=25)
    plt.subplots_adjust(bottom=0.3)
    # 保存
    mkdir(save_fig_dir)
    plt.savefig(os.path.join(
        save_fig_dir, 'compare_methods.png'), bbox_inches='tight')
    plt.close()


def get_Automatoes(args, stems):
    balance, economic = get_sim_res(
        stems, args.wur_team_files, args.base_tmp_folder, args.version)
    return balance, economic


def get_sim_res(stems, trainDir, base_tmp_folder, version):
    tmp_folder = os.path.join(base_tmp_folder, 'models/%s' % version)

    wur_tomato_reader = TomatoDataset(trainDir, tmp_folder)
    train_data = wur_tomato_reader.read_data(trainDir)
    full_train_x, _ = wur_tomato_reader.data_process(train_data)

    period_action = full_train_x[0, :, 6:10]

    ten_env = env(version, base_tmp_folder)

    X = np.concatenate((period_action, period_action[-49:, :]), axis=0)
    balance, TeamsEconomic = simOurModel(
        X, ten_env, stems)

    return balance, TeamsEconomic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input_path", default="./input", type=str)
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--wur_team_files",
                        default="./input/team.txt", type=str)
    parser.add_argument("--wur_champion_files",
                        default="./input/wur_champion.txt", type=str)
    parser.add_argument("--DAY_IN_LIFE_CYCLE",
                        default=166, type=int)
    parser.add_argument("--version", default="incremental", type=str)
    parser.add_argument("--GA_train_result_path",
                        default="GA/ga_train/policy/", type=str)
    parser.add_argument("--sac_model_dir",
                        default="SAC/sac_model/", type=str)

    parser = ga_params(parser)
    parser = sac_params(parser)

    args = parser.parse_args()

    Figure4(args)
