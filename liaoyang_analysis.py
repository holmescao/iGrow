'''
Author: your name
Date: 2021-06-12 23:12:56
LastEditTime: 2021-06-28 21:15:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NMI/liaoyang_analysis.py
'''
import matplotlib as mpl
import seaborn as sns
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from utils.plt_params import plt_fig_params
from utils.common import mkdir

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'


def Plot_Pair(args, df, file_name):
    mpl.rcParams.update(plt_fig_params)
    plt.figure(figsize=(13, 13))
    sns.set(style='ticks', font_scale=1.5)
    g = sns.pairplot(df,
                     kind='scatter', diag_kind='kde', corner=True,
                     hue='method',
                     markers=['o', 'D'],
                     palette=sns.color_palette("viridis_r", 2),
                     plot_kws={
                         "s": 25,
                          "alpha": 0.3,
                          'lw': 0.1,
                     },
                     x_vars=['AirT', 'AirCO2', 'AirPAR', 'AirRH'],
                     y_vars=['AirT', 'AirCO2', 'AirPAR', 'AirRH'])

    handles = g._legend_data.values()
    # labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=['the control group', 'the experimental group'],
                 bbox_to_anchor=(1, 1),
                 loc='upper right', ncol=1, fontsize=22, framealpha=0.5)
    g._legend.remove()
    g.fig.subplots_adjust(bottom=0.9)
    plt.subplots_adjust(bottom=0.4)
    sns.set_style('ticks', {'axes.grid': True})
    sns.despine(top=False, right=False, left=False, bottom=False,
                offset=None, trim=False)
    mkdir(args.fig_save_dir)
    plt.savefig(args.fig_save_dir+file_name)
    plt.close()


def Figure6(args):
    for control in args.control_group:
        for experiment in args.experiment_gh:
            file_name = 'sensor_interval@1h_expr@%d_ctrl@%d' % (
                experiment, control)
            save_path = args.sensor_save_path + file_name + '.csv'
            sensor_data = pd.read_csv(save_path)

            Plot_Pair(args, sensor_data, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_gh', type=list, default=[14, 15, 25, 27, 28],
                        help='ids of all green house.')
    parser.add_argument('--control_group', type=list, default=[7, 13],
                        help='ids of all green house.')
    parser.add_argument('--fig_save_dir', type=str, default='result/figure6/gh_sensor/',
                        help='directory to save figures.')
    parser.add_argument('--sensor_save_path', type=str, default='data/liaoyang2_sensor/',
                        help='directory to save sensor data.')
    args = parser.parse_args()

    Figure6(args)
