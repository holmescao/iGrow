import argparse
import os
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import torch
import gym

from TenSim.utils.data_reader import TomatoDataset
from TenSim.simulator_gpu import PredictModel
from utils.common import mkdir, save_curve
from utils.plt_params import plt_fig_params, set_day_xtick

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
gym.logger.set_level(40)
torch.set_num_threads(1)


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


def get_stems(files_path):
    with open(files_path, "r") as f:
        data = f.readlines()
    for line in data:
        if "stems" in line:
            stems = np.load(line.strip("\n"))
            break
    return stems


def Figure3(args):
    print("=============Figure3===============")
    save_dir = args.base_tmp_folder+'/figure3/'
    mkdir(save_dir)

    stems = get_stems(args.wur_champion_files)

    _, Baseline = get_sim_res(
        stems, args.wur_team_files, args.base_tmp_folder, version='baseline')
    _, Incremental = get_sim_res(
        stems, args.wur_team_files, args.base_tmp_folder, version='incremental')

    with open(args.wur_champion_files, 'r') as f:
        wur_champion_file_list = f.readlines()
    WUREconomic = pd.read_csv(wur_champion_file_list[0].replace("\n", ""))
    df = pd.read_excel(wur_champion_file_list[1].replace("\n", ""))
    real = df['Unnamed: 12'].iloc[38:38+args.DAY_IN_LIFE_CYCLE].values

    plantcost = 4.29

    # save
    fig3a_baseline = {"xlabel": 'date', "ylabel": "euro/m2",
                      "x": np.arange(len(Baseline['balance'])),
                      "y": np.cumsum(Baseline['balance'])-plantcost}
    fig3a_incremental = {"xlabel": 'date', "ylabel": "euro/m2",
                         "x": np.arange(len(Incremental['balance'])),
                         "y": np.cumsum(Incremental['balance'])-plantcost}
    fig3a_wursim = {"xlabel": 'date', "ylabel": "euro/m2",
                    "x": np.arange(len(WUREconomic)),
                    "y": WUREconomic['balance'].values-plantcost}
    fig3a_real = {"xlabel": 'date', "ylabel": "euro/m2",
                  "x": np.arange(len(real)),
                  "y": real.astype(np.float32)}
    save_curve_dir = save_dir + '/curve/'
    mkdir(save_curve_dir)
    save_curve(fig3a_baseline, save_curve_dir+'fig3a_baseline.pkl')
    save_curve(fig3a_incremental, save_curve_dir+'fig3a_incremental.pkl')
    save_curve(fig3a_wursim, save_curve_dir+'fig3a_wursim.pkl')
    save_curve(fig3a_real, save_curve_dir+'fig3a_real.pkl')

    show_figure3(fig3a_baseline['y'],
                 fig3a_incremental['y'],
                 fig3a_wursim['y'],
                 fig3a_real['y'],
                 days=args.DAY_IN_LIFE_CYCLE,
                 startDate='2019-12-16', endDate='2020-05-29',
                 save_fig_dir=save_dir)


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


def show_figure3(baseline,
                 incremental,
                 wursim,
                 real,
                 days,
                 startDate,
                 endDate,
                 save_fig_dir):
    # fig, axes
    mpl.rcParams.update(plt_fig_params)

    # 更新参数
    props = {0: {"ylabel": "euro / m$^2$", "ylim": [-25, 10]},
             1: {"ylabel": "error", }}
    x_ticks_interval = {0: 4, 1: 4, 2: 4}

    curve_name = ['Groundtruth', 'Wur simulator',
                  'Baseline model', 'Incremental model']
    color_map = {curve_name[0]: cm.autumn(0),
                 curve_name[1]: cm.viridis(0.6),
                 curve_name[2]: cm.cool(0.3),
                 curve_name[3]: cm.winter(0.3)}
    plt_fig_style = {
        curve_name[0]: dict(linestyle='--', lw=1.5, color=color_map[curve_name[0]], label=curve_name[0]),
        curve_name[1]: dict(linestyle='-', lw=1.5, color=color_map[curve_name[1]], label=curve_name[1]),
        curve_name[2]: dict(linestyle='-', lw=1.5, color=color_map[curve_name[2]], label=curve_name[2]),
        curve_name[3]: dict(linestyle='-', lw=1.5, color=color_map[curve_name[3]], label=curve_name[3])}

    names = ['(a) NetProfit', '(b) Accumulative absolute error']

    for i in range(2):
        fig = plt.figure(figsize=(8, 4))
        layout = (1, 1)
        for c in range(0, layout[1], 1):
            plt.subplot2grid(layout, (0, c), rowspan=1, colspan=1)
        ax = fig.axes[0]

        if i == 0:
            ax.plot(real[:days], **plt_fig_style[curve_name[0]])
            ax.plot(wursim[:days],
                    **plt_fig_style[curve_name[1]])
            ax.plot(baseline[:days],
                    **plt_fig_style[curve_name[2]])
            ax.plot(incremental[:days],
                    **plt_fig_style[curve_name[3]])

            ax.set_yticks(ticks=list(range(-20, 20, 10)))
        else:
            our_real = np.cumsum(np.abs(incremental[:days] - real[:days]))
            our_noreal = np.cumsum(np.abs(baseline[:days] - real[:days]))
            wur_sim = np.cumsum(np.abs(wursim[:days] - real[:days]))

            alpha = 0.8
            ax.stackplot(np.arange(len(real[:days])), our_noreal,
                         labels=[curve_name[2]],
                         color=color_map[curve_name[2]],
                         edgecolor='k',
                         linewidth=2,
                         alpha=alpha,
                         baseline='zero')
            ax.stackplot(np.arange(len(real[:days])), our_real,
                         labels=[curve_name[3]],
                         color=color_map[curve_name[3]],
                         edgecolor='k',
                         linewidth=2,
                         alpha=alpha,
                         baseline='zero')
            ax.stackplot(np.arange(len(real[:days])), wur_sim,
                         labels=[curve_name[1]],
                         color=color_map[curve_name[1]],
                         edgecolor='k',
                         linewidth=2,
                         alpha=alpha,
                         baseline='zero')

            ax.set_yticks(ticks=list(range(100, 500, 100)))

        ax.set(**props[i])
        ax.set_xticklabels(labels='date')
        ax.set_title(names[i], y=-0.3, fontsize=20)
        ticks, labels = set_day_xtick(
            x_ticks_interval[i], list(real[:days]), startDate, endDate)
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=labels)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        min_xlim, max_xlim = ax.get_xlim()
        min_ylim, max_ylim = ax.get_ylim()
        xlim_length = abs(max_xlim - min_xlim)
        ylim_length = abs(max_ylim - min_ylim)
        aspect = xlim_length / ylim_length
        ax.set_aspect(aspect*1)

        ax.grid(linestyle="--", alpha=0.4)

        plt.tight_layout()

        # legend
        ax = fig.axes[0]
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.23), loc='upper left',
        #           ncol=2, framealpha=0, fancybox=False, fontsize=20)
        # plt.subplots_adjust(left=0.1, bottom=0.9, right=0.95,
        #                     top=1, wspace=0.4)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.25), loc='upper center',
                  ncol=2, framealpha=0, fancybox=False, fontsize=16)

        mkdir(save_fig_dir)
        plt.savefig(os.path.join(save_fig_dir, 'figure3_(%d).png' % (i+1)),
                    bbox_inches='tight')
        plt.close()


def get_sim_res(stems, trainDir, base_tmp_folder, version):
    # tensim version
    tmp_folder = os.path.join(base_tmp_folder, 'models/%s' % version)

    wur_tomato_reader = TomatoDataset(trainDir, tmp_folder)
    train_data = wur_tomato_reader.read_data(trainDir)
    full_train_x, _ = wur_tomato_reader.data_process(train_data)

    period_action = full_train_x[0, :, 6:10]
    X = np.concatenate((period_action, period_action[-49:, :]), axis=0)

    ten_env = env(version, base_tmp_folder)
    balance, TeamsEconomic = simOurModel(X, ten_env, stems)

    return balance, TeamsEconomic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gh", default="Automatoes", type=str)
    parser.add_argument("--base_input_path", default="./input", type=str)
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--wur_team_files",
                        default="./input/team.txt", type=str)
    parser.add_argument("--wur_champion_files",
                        default="./input/wur_champion.txt", type=str)
    parser.add_argument("--DAY_IN_LIFE_CYCLE",
                        default=166, type=int)
    args = parser.parse_args()

    Figure3(args)
