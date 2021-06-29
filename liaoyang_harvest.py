import argparse
import os
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import datetime
import seaborn as sns

from utils.common import mkdir, save_curve
from utils.plt_params import plt_fig_params, set_day_xtick

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'


def Figure5(args):
    print("=============Figure5===============")
    save_dir = args.base_tmp_folder+'/figure5/'
    mkdir(save_dir)

    expr_harvest, ctrl_harvest = get_harvest(args)
    price = get_price(args)

    # save curve
    figure5a_experimental = {"xlabel": 'date', "ylabel": "kg/m2",
                             "x": range(len(expr_harvest['production'])),
                             "y": expr_harvest['production']}
    figure5a_control = {"xlabel": 'date', "ylabel": "kg/m2",
                        "x": range(len(ctrl_harvest['production'])),
                        "y": ctrl_harvest['production']}
    figure5b_experimental = {"xlabel": 'date', "ylabel": "euro/m2",
                             "x": range(len(expr_harvest['gains'])),
                             "y": expr_harvest['gains']}
    figure5b_control = {"xlabel": 'date', "ylabel": "euro/m2",
                        "x": range(len(ctrl_harvest['gains'])),
                        "y": ctrl_harvest['gains']}
    figure5c_experimental = {"xlabel": ['Growing expert', 'iGrow'], "ylabel": "euro",
                             "x": range(len(price['expr'])),
                             "y": price['expr']}
    figure5c_control = {"xlabel": ['Growing expert', 'iGrow'], "ylabel": "euro",
                        "x": range(len(price['ctrl'])),
                        "y": price['ctrl']}

    save_curve_dir = save_dir + '/curve/'
    mkdir(save_curve_dir)
    save_curve(figure5a_experimental, save_curve_dir +
               'figure5a_experimental.pkl')
    save_curve(figure5a_control, save_curve_dir+'figure5a_control.pkl')
    save_curve(figure5b_experimental, save_curve_dir +
               'figure5b_experimental.pkl')
    save_curve(figure5b_control, save_curve_dir+'figure5b_control.pkl')
    save_curve(figure5c_experimental, save_curve_dir +
               'figure5c_experimental.pkl')
    save_curve(figure5c_control, save_curve_dir+'figure5c_control.pkl')
    # show
    compare_harvest_plot(expr_harvest, ctrl_harvest, price,
                         startDate=args.startDate,
                         endDate=args.endDate,
                         save_fig_dir=args.save_fig_dir)


def get_harvest(args):
    harvest_file = os.path.join(args.base_input_path, args.harvest_files)
    with open(harvest_file, 'r') as f:
        harvest_file_dir = f.readlines()
    harvest_file_dir = harvest_file_dir[0].replace("\n", '')

    expr_harvest, ctrl_harvest = harvest_analysis(args=args,
                                                  harvest_dir=harvest_file_dir)

    return expr_harvest, ctrl_harvest


def compare_harvest_plot(expr_harvest, ctrl_harvest, df,
                         startDate, endDate,
                         save_fig_dir):
    # fig, axes
    mpl.rcParams.update(plt_fig_params)
    fig = plt.figure(figsize=(13, 6))
    layout = (1, 3)
    for c in range(layout[1]):
        plt.subplot2grid(layout, (0, c), rowspan=1, colspan=1)

    # 更新参数
    props = {0: {"xlabel": "date",
                 "ylabel": "kg / m2",
                 },
             1: {"xlabel": "date",
                 "ylabel": "euro / m2",
                 },
             2: {"ylabel": "euro",
                 },
             }
    plt_fig_style = {
        'Human expert': dict(linestyle='--', lw=2, color=cm.viridis(0.7), label='the control group'),
        'EGA': dict(linestyle='-', lw=2, color=cm.viridis(0.3), label='the experimental group'), }

    method = list(plt_fig_style.keys())
    title = ['(a) Production', '(b) Gains', '(c) Price']
    key = ['production', 'gains', 'price']

    yticks_list = [list(range(0, 20, 5)) + [20],
                   list(range(0, 10, 3))]
    # draw基础类对象
    for idx, ax in enumerate(fig.axes):
        if idx == 2:
            columns = df.columns
            sns.boxplot(data=df, notch=0, linewidth=1.5,
                        order=list(columns), dodge=True, width=0.6,
                        palette=sns.color_palette("viridis_r", 2))
            # 美化
            ax.set_xticklabels(labels=['Growing expert', 'iGrow'])
            ax.tick_params(axis='x', labelsize=22)
        else:
            experiment_avg = np.mean(expr_harvest[key[idx]], axis=1)
            experiment_std = np.std(expr_harvest[key[idx]], axis=1)
            control_avg = np.mean(ctrl_harvest[key[idx]], axis=1)
            control_std = np.std(ctrl_harvest[key[idx]], axis=1)

            iter = np.arange(len(experiment_avg))

            ax.plot(iter, experiment_avg, **plt_fig_style['EGA'])
            r1 = list(map(lambda x: x[0] - x[1],
                          zip(experiment_avg, experiment_std)))
            r2 = list(map(lambda x: x[0] + x[1],
                          zip(experiment_avg, experiment_std)))
            ax.fill_between(iter, r1, r2, alpha=0.3, **plt_fig_style['EGA'])

            ax.plot(iter, control_avg, **plt_fig_style['Human expert'])
            r1 = list(map(lambda x: x[0] - x[1],
                          zip(control_avg, control_std)))
            r2 = list(map(lambda x: x[0] + x[1],
                          zip(control_avg, control_std)))
            ax.fill_between(iter, r1, r2, alpha=0.3, **
                            plt_fig_style['Human expert'])

            # 美化
            xticks, xlabels = set_day_xtick(num=4,
                                            var_list=list(experiment_avg[:]),
                                            startDate=startDate,
                                            endDate=endDate)

            ax.set_xticks(ticks=xticks)
            ax.set_xticklabels(labels=xlabels)
            ax.set_yticks(ticks=yticks_list[idx])

            ax.tick_params(axis='x', labelsize=20)
            ax.grid(linestyle="--", alpha=0.4)

        ax.set_title(title[idx], y=-0.34, fontsize=25)
        ax.tick_params(axis='y', labelsize=20)

        ax.set(**props[idx])  # 参数设置

        min_xlim, max_xlim = ax.get_xlim()
        min_ylim, max_ylim = ax.get_ylim()
        xlim_length = abs(max_xlim - min_xlim)
        ylim_length = abs(max_ylim - min_ylim)
        aspect = xlim_length / ylim_length
        ax.set_aspect(aspect)

        ax.xaxis.label.set_size(25)
        ax.yaxis.label.set_size(25)

    plt.tight_layout()

    # legend
    ax = fig.axes[1]
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], bbox_to_anchor=(0.7, -0.3), loc='upper right',
               ncol=2, framealpha=0, fancybox=False, fontsize=30)
    plt.subplots_adjust(bottom=0.4)

    # 保存
    mkdir(save_fig_dir)
    plt.savefig(save_fig_dir+'liaoyang2_harvest.png', bbox_inches='tight')
    plt.close()


def harvest_analysis(args, harvest_dir):
    # 结果初始化
    startDate = datetime.datetime.strptime(args.startDate, "%Y-%m-%d")
    endDate = datetime.datetime.strptime(args.endDate, "%Y-%m-%d")
    days = (endDate-startDate).days + 1
    expr_prod = np.zeros((days, len(args.experiment_gh)))
    ctrl_prod = np.zeros((days, len(args.control_group)))
    expr_gains = np.zeros((days, len(args.experiment_gh)))
    ctrl_gains = np.zeros((days, len(args.control_group)))
    # 读取收成数据文件
    m2_to_Mu = 667
    production = pd.read_csv(harvest_dir + 'production.csv')
    production = production.values[:, 1:] / m2_to_Mu
    Income = pd.read_csv(harvest_dir + 'Income.csv')
    Income = Income.values[:, 1:] / m2_to_Mu * args.rmb2euro

    ctrl_prod[-len(production):, :] = np.nancumsum(production[:, :2], axis=0)
    expr_prod[-len(production):, :] = np.nancumsum(production[:, 2:], axis=0)

    ctrl_gains[-len(Income):, :] = np.nancumsum(Income[:, :2], axis=0)
    expr_gains[-len(Income):, :] = np.nancumsum(Income[:, 2:], axis=0)

    expr_harvest = {"production": expr_prod,
                    "gains": expr_gains}
    ctrl_harvest = {"production": ctrl_prod,
                    "gains": ctrl_gains}

    return expr_harvest, ctrl_harvest


def scatter_data(df, col, pos_x):
    expr = df[col].values
    expr_dic = dict(zip(*np.unique(expr, return_counts=True)))
    Y = []
    Val = []
    for k, v in expr_dic.items():
        if k != 'nan':
            Y.append(float(k))
            Val.append(v)
    X = [pos_x] * len(Y)

    X = np.array(X)
    Y = np.array(Y)
    Val = np.array(Val)

    return X, Y, Val


def get_price(args):
    harvest_file = os.path.join(args.base_input_path, args.harvest_files)
    with open(harvest_file, 'r') as f:
        harvest_file_dir = f.readlines()
    harvest_file_dir = harvest_file_dir[0].replace("\n", '')
    harvest_price_dir = os.path.join(harvest_file_dir, 'price.csv')
    df = pd.read_csv(harvest_price_dir)

    ctrl_price = df.values[:, 1:3]
    expr_price = df.values[:, 3:]
    expr_price = expr_price.astype(np.float32) * args.rmb2euro
    ctrl_price = ctrl_price.astype(np.float32) * args.rmb2euro

    expr_price[expr_price == 0] = np.nan
    ctrl_price[ctrl_price == 0] = np.nan

    expr_price = expr_price.flatten()
    ctrl_price = ctrl_price.flatten()
    price = np.full((expr_price.shape[0], 2), np.nan)

    columns = ['ctrl', 'expr']
    price[:ctrl_price.shape[0], 0] = ctrl_price
    price[:, 1] = expr_price
    df = pd.DataFrame(price, columns=columns)
    df = df.applymap("{0:.01f}".format)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--startDate', default="2020-03-15",
                        help='start date of planting.')
    parser.add_argument('--endDate', default="2020-07-13",
                        help='end date of planting.')
    parser.add_argument('--experiment_gh', type=list, default=[14, 15, 25, 27, 28],
                        help='ids of all green house.')
    parser.add_argument('--control_group', type=list, default=[7, 13],
                        help='ids of all green house.')
    parser.add_argument('--rmb2euro', type=float, default=0.1276,
                        help="rate of rmb to euro")
    parser.add_argument("--base_input_path", default="./input", type=str)
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--harvest_files", default='harvest.txt', type=str)
    parser.add_argument('--save_fig_dir', type=str, default='result/figure5/',
                        help="save figures directory")
    args = parser.parse_args()

    Figure5(args)
