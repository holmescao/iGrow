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
from TenSim.simulator import PredictModel
from utils.common import mkdir, save_curve, load_curve
from utils.plt_params import plt_fig_params, set_day_xtick, set_ytick
from GA.ga_module.config import setting_test
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


def SaveCurve(EGA_economic, SAC_economic, Automatoes_economic, figure4d_EGA_lmp_control, figure4d_Automatoes_lmp_control, save_curve_dir):

    plantcost = 4.29
    price = 3.185
    # figure 4(a)
    figure4a_production_EGA = np.cumsum(EGA_economic['gains']) / price
    figure4a_production_EGA_dict = {"xlabel": 'Date', "ylabel": "Kg/m$^2$",
                                    "x": range(len(figure4a_production_EGA)),
                                    "y": figure4a_production_EGA}
    figure4a_production_SAC = np.cumsum(SAC_economic['gains']) / price
    figure4a_production_SAC_dict = {"xlabel": 'Date', "ylabel": "Kg/m$^2$",
                                    "x": range(len(figure4a_production_SAC)),
                                    "y": figure4a_production_SAC}
    figure4a_production_Automatoes = np.cumsum(
        Automatoes_economic['gains']) / price
    figure4a_production_Automatoes_dict = {"xlabel": 'Date', "ylabel": "Kg/m$^2$",
                                           "x": range(len(figure4a_production_Automatoes)),
                                           "y": figure4a_production_Automatoes}
    # figure 4(b)
    figure4b_cost_EGA = np.cumsum(EGA_economic['variableCosts'])
    figure4b_cost_EGA_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                              "x": range(len(figure4b_cost_EGA)),
                              "y": figure4b_cost_EGA}
    figure4b_cost_SAC = np.cumsum(SAC_economic['variableCosts'])
    figure4b_cost_SAC_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                              "x": range(len(figure4b_cost_SAC)),
                              "y": figure4b_cost_SAC}
    figure4b_cost_Automatoes = np.cumsum(Automatoes_economic['variableCosts'])
    figure4b_cost_Automatoes_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                                     "x": range(len(figure4b_cost_Automatoes)),
                                     "y": figure4b_cost_Automatoes}
    # figure 4(c)
    figure4c_balance_EGA = np.cumsum(EGA_economic['balance'])-plantcost
    figure4c_balance_EGA_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                                 "x": range(len(figure4c_balance_EGA)),
                                 "y": figure4c_balance_EGA}
    figure4c_balance_SAC = np.cumsum(SAC_economic['balance'])-plantcost
    figure4c_balance_SAC_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                                 "x": range(len(figure4c_balance_SAC)),
                                 "y": figure4c_balance_SAC}
    figure4c_balance_Automatoes = np.cumsum(
        Automatoes_economic['balance'])-plantcost
    figure4c_balance_Automatoes_dict = {"xlabel": 'Date', "ylabel": "Euro/m$^2$",
                                        "x": range(len(figure4c_balance_Automatoes)),
                                        "y": figure4c_balance_Automatoes}
    # figure 4(d)
    figure4d_EGA_lmp_control_dict = {"xlabel": 'Hour', "ylabel": "On=1,Off=0",
                                     "x": np.arange(len(figure4d_EGA_lmp_control)),
                                     "y": figure4d_EGA_lmp_control}
    figure4d_Automatoes_lmp_control_dict = {"xlabel": 'Hour', "ylabel": "On=1,Off=0",
                                            "x": np.arange(len(figure4d_Automatoes_lmp_control)),
                                            "y": figure4d_Automatoes_lmp_control}

    # figure 4(a)
    save_curve(figure4a_production_EGA_dict, save_curve_dir +
               'figure4a_production_EGA_dict.pkl')
    save_curve(figure4a_production_SAC_dict, save_curve_dir +
               'figure4a_production_SAC_dict.pkl')
    save_curve(figure4a_production_Automatoes_dict, save_curve_dir +
               'figure4a_production_Automatoes_dict.pkl')
    # figure 4(b)
    save_curve(figure4b_cost_EGA_dict, save_curve_dir +
               'figure4b_cost_EGA_dict.pkl')
    save_curve(figure4b_cost_SAC_dict, save_curve_dir +
               'figure4b_cost_SAC_dict.pkl')
    save_curve(figure4b_cost_Automatoes_dict, save_curve_dir +
               'figure4b_cost_Automatoes_dict.pkl')
    # figure 4(c)
    save_curve(figure4c_balance_EGA_dict, save_curve_dir +
               'figure4c_balance_EGA_dict.pkl')
    save_curve(figure4c_balance_SAC_dict, save_curve_dir +
               'figure4c_balance_SAC_dict.pkl')
    save_curve(figure4c_balance_Automatoes_dict, save_curve_dir +
               'figure4c_balance_Automatoes_dict.pkl')
    # figure 4(d)
    save_curve(figure4d_EGA_lmp_control_dict, save_curve_dir +
               'figure4d_EGA_lmp_control_dict.pkl')
    save_curve(figure4d_Automatoes_lmp_control_dict, save_curve_dir +
               'figure4d_Automatoes_lmp_control_dict.pkl')

    # show
    sim_res = {
        "figure4a": {
            "values": {
                "EGA": figure4a_production_EGA,
                "SAC": figure4a_production_SAC,
                "Automatoes": figure4a_production_Automatoes},
            "props": {"xlabel": 'Date', "ylabel": "Kg/m$^2$"}},
        "figure4b": {
            "values": {
                "EGA": figure4b_cost_EGA,
                "SAC": figure4b_cost_SAC,
                "Automatoes": figure4b_cost_Automatoes},
            "props": {"xlabel": 'Date', "ylabel": "Euro/m$^2$"}},
        "figure4c": {
            "values": {
                "EGA": figure4c_balance_EGA,
                "SAC": figure4c_balance_SAC,
                "Automatoes": figure4c_balance_Automatoes},
            "props": {"xlabel": 'Date', "ylabel": "Euro/m$^2$"}},
        "figure4d": {
            "values": {
                "EGA": figure4d_EGA_lmp_control,
                "Automatoes": figure4d_Automatoes_lmp_control},
            "props": {"xlabel": 'Hour', "ylabel": "on=1,off=0", }
        },
    }

    return sim_res


def LoadCurve(save_curve_dir):
    # figure 4(a)
    figure4a_production_EGA_dict = load_curve(save_curve_dir +
                                              'figure4a_production_EGA_dict.pkl')
    figure4a_production_SAC_dict = load_curve(save_curve_dir +
                                              'figure4a_production_SAC_dict.pkl')
    figure4a_production_Automatoes_dict = load_curve(save_curve_dir +
                                                     'figure4a_production_Automatoes_dict.pkl')
    # figure 4(b)
    figure4b_cost_EGA_dict = load_curve(save_curve_dir +
                                        'figure4b_cost_EGA_dict.pkl')
    figure4b_cost_SAC_dict = load_curve(save_curve_dir +
                                        'figure4b_cost_SAC_dict.pkl')
    figure4b_cost_Automatoes_dict = load_curve(save_curve_dir +
                                               'figure4b_cost_Automatoes_dict.pkl')
    # figure 4(c)
    figure4c_balance_EGA_dict = load_curve(save_curve_dir +
                                           'figure4c_balance_EGA_dict.pkl')
    figure4c_balance_SAC_dict = load_curve(save_curve_dir +
                                           'figure4c_balance_SAC_dict.pkl')
    figure4c_balance_Automatoes_dict = load_curve(save_curve_dir +
                                                  'figure4c_balance_Automatoes_dict.pkl')
    # figure 4(d)
    figure4d_EGA_lmp_control_dict = load_curve(save_curve_dir +
                                               'figure4d_EGA_lmp_control_dict.pkl')
    figure4d_Automatoes_lmp_control_dict = load_curve(save_curve_dir +
                                                      'figure4d_Automatoes_lmp_control_dict.pkl')

    # show
    sim_res = {
        "figure4a": {
            "values": {
                "EGA": figure4a_production_EGA_dict['y'],
                "SAC": figure4a_production_SAC_dict['y'],
                "Automatoes": figure4a_production_Automatoes_dict['y'], },
            "props": {"xlabel": 'Date', "ylabel": "Kg/m$^2$"}},
        "figure4b": {
            "values": {
                "EGA": figure4b_cost_EGA_dict['y'],
                "SAC": figure4b_cost_SAC_dict['y'],
                "Automatoes": figure4b_cost_Automatoes_dict['y'], },
            "props": {"xlabel": 'Date', "ylabel": "Euro/m$^2$"}},
        "figure4c": {
            "values": {
                "EGA": figure4c_balance_EGA_dict['y'],
                "SAC": figure4c_balance_SAC_dict['y'],
                "Automatoes": figure4c_balance_Automatoes_dict['y'], },
            "props": {"xlabel": 'Date', "ylabel": "Euro/m$^2$"}},
        "figure4d": {
            "values": {
                "EGA": figure4d_EGA_lmp_control_dict['y'],
                "Automatoes": figure4d_Automatoes_lmp_control_dict['y'], },
            "props": {"xlabel": 'Hour', "ylabel": "On=1,Off=0", }
        },
    }

    return sim_res


def Figure4(args):
    print("=============Figure4===============")
    save_dir = args.base_tmp_folder+'/figure4/'
    if not os.path.exists(save_dir):
        mkdir(save_dir)
    save_curve_dir = save_dir + '/curve/'

    if os.path.exists(save_curve_dir):
        sim_res = LoadCurve(save_curve_dir)
    else:
        stems = get_stems(args.wur_champion_files)

        # curve
        EGA_economic = get_EGA(args, stems)
        SAC_economic = get_SAC(args, stems)
        Automatoes_economic = get_Automatoes(args, stems)
        figure4d_EGA_lmp_control = get_control(args, method='EGA', var='lmp')
        figure4d_Automatoes_lmp_control = get_control(
            args, method='Automatoes', var='lmp')

        # save
        mkdir(save_curve_dir)
        sim_res = SaveCurve(EGA_economic,
                            SAC_economic,
                            Automatoes_economic,
                            figure4d_EGA_lmp_control,
                            figure4d_Automatoes_lmp_control,
                            save_curve_dir)

    compare_plot(sim_res=sim_res,
                 startDate=args.startDate,
                 endDate=args.endDate,
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
        setpoints = list(map(round, link[2]))

    return setpoints


def get_EGA(args, stems):
    economic = ga_sim(args, stems)

    return economic


def get_SAC(args, stems):
    _, economic = sac_sim(args, stems)

    return economic


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
    _, economic = simOurModel(policy, ten_env, stems)

    return economic


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

    while not done:
        a = period_action[day*dims: (day+1)*dims, :]

        a = a.reshape((-1), order='F')
        _, r, done, ec = ten_env.step(a)
        day += 1
        reward.append(float(r))

        for k, v in ec.items():
            economic[k].append(v)

    return reward, economic


def ga_params(parser):
    parser.add_argument(
        '--seed', default=setting_test['params_info']['seed'], help='random seed')
    parser.add_argument(
        '--NIND', default=setting_test['params_info']['NIND'], help='population size')
    parser.add_argument(
        '--MAXGEN', default=setting_test['params_info']['MAXGEN'], help='maximum generation')
    parser.add_argument(
        '--LINKDAY', default=setting_test['params_info']['LINKDAY'], help='copy days')
    parser.add_argument(
        '--XOVR', default=setting_test['params_info']['XOVR'], help='crossover probability')

    return parser


def sac_params(parser):
    parser.add_argument('--sac_actor', default="sac_actor_7900_Exp")
    parser.add_argument('--sac_critic', default="sac_critic_7900_Exp")

    return parser


def compare_plot(sim_res, startDate, endDate, save_fig_dir):
    # fig, axes
    mpl.rcParams.update(plt_fig_params)
    fig = plt.figure(figsize=(13, 8))
    layout = (2, 2)
    for r in range(layout[0]):
        for c in range(layout[1]):
            plt.subplot2grid(layout, (r, c), rowspan=1, colspan=1)

    colors = {"Automatoes": '#ec4646',
              "EGA": cm.viridis(0.3),
              "SAC": cm.viridis(0.6)}
    lw = 2.5
    plt_fig_style = {
        'Automatoes': dict(linestyle='--', lw=lw, alpha=1, color=colors['Automatoes'], label='Automatoes'),
        'EGA': dict(linestyle='-', lw=lw, alpha=1, color=colors["EGA"], label='EGA'),
        'SAC': dict(linestyle='-', lw=lw, alpha=1, color=colors["SAC"], label='SAC'), }

    sub_titles = ['(a) Crop yield', '(b) Cost',
                  '(c) NetProfit', '(d) Illumination action']
    fig_ax = dict(zip(sim_res.keys(), range(len(sim_res.keys()))))
    yticks_num = dict(zip(sim_res.keys(), [4]*3+[2]))

    for sub_fig, info in sim_res.items():
        values = info['values']
        props = info['props']
        ax_id = fig_ax[sub_fig]

        ax = fig.axes[ax_id]

        max_val, min_val = 0, 0
        for method, val in values.items():
            ax.plot(val, **plt_fig_style[method])
            max_val = max(val) if max(val) > max_val else max_val
            min_val = min(val) if min(val) < min_val else min_val

        if sub_fig == "figure4d":
            xticks = list(range(0, 120, 30)) + [120]
            ax.set_xticks(ticks=xticks)
        else:
            xticks, xlabels = set_day_xtick(num=4,
                                            var_list=val,
                                            startDate=startDate,
                                            endDate=endDate)
            ax.set_xticks(ticks=xticks)
            ax.set_xticklabels(labels=xlabels)

        yticks = set_ytick(num=yticks_num[sub_fig],
                           max_val=max_val,
                           min_val=min_val)
        ax.set_yticks(ticks=yticks)

        ax.set_title(sub_titles[ax_id], y=-0.4, fontsize=25)

        min_xlim, max_xlim = ax.get_xlim()
        min_ylim, max_ylim = ax.get_ylim()
        xlim_length = abs(max_xlim - min_xlim)
        ylim_length = abs(max_ylim - min_ylim)
        aspect = xlim_length / ylim_length
        ax.set_aspect(aspect*0.9)

        ax.set(**props)
        ax.grid(linestyle="--", alpha=0.4)

    # legend
    # ax = fig.axes[1]
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left',
    #           ncol=3, framealpha=0, fancybox=False, fontsize=25)
    # plt.subplots_adjust(bottom=0.6)
    plt.tight_layout()

    mkdir(save_fig_dir)
    plt.savefig(os.path.join(
        save_fig_dir, 'compare_methods.png'))
    # plt.savefig(os.path.join(
    #     save_fig_dir, 'compare_methods.png'), bbox_inches='tight')
    plt.close()


def get_Automatoes(args, stems):
    balance, economic = get_sim_res(
        stems, args.wur_team_files, args.base_tmp_folder, args.version)
    return economic


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
    parser.add_argument("--startDate", default="2019-12-16",
                        help="start date of planting",)
    parser.add_argument("--endDate", default="2020-05-29",
                        help="end date of planting")

    parser = ga_params(parser)
    parser = sac_params(parser)

    args = parser.parse_args()

    Figure4(args)
