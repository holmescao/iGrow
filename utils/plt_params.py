import numpy as np
import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl


plt_fig_params = {
    'agg.path.chunksize': 10000,
    'figure.autolayout': True,
    "figure.dpi": 150,
    # 'figure.figsize': [10, 12],
    # 'figure.figsize': [10, 6],
    'figure.figsize': [6, 6],
    # 'figure.figsize': [15, 8],
    "font.size": 18,
    'font.family': 'STIXGeneral',
    "figure.subplot.wspace": 0.2,

    "figure.subplot.hspace": 0.4,

    "axes.spines.right": False,

    "axes.spines.top": False,

    "axes.titlesize": 22,

    "axes.labelsize": 18,

    "legend.fontsize": 18,

    'savefig.pad_inches': 0,
    "xtick.labelsize": 16,

    "ytick.labelsize": 16,

    "xtick.direction": 'in',

    "ytick.direction": 'in'

}

plt_fig_props = {"xlabel": "xlabel",
                 "ylabel": "ylabel",
                 "title": "title",
                 "xticks": "xticks",
                 "yticks": "yticks"}

plt_fig_style = {
    'default': dict(linestyle='-', lw=1.5, color=cm.hsv(0.2)),
    'WT_Price': dict(linestyle='-', marker='*', markersize=6, color='#d7191c'),
    'DT_Price': dict(linestyle='--', marker='s', markersize=6, color='#abdda4'),
    'TT_Price': dict(linestyle='-.', marker='v', markersize=6, color='#2b83ba')
}


def beauty_plot():
    # fig, axes
    layout = (3, 3)
    fig, axes = plt.sublpots(*layout)

    mpl.rcParams.update(params)
    props = {"xlabel": "days",
             "ylabel": "euro/m2"}

    for i, ax in enumerate(axes.flat):
        key = 'AICU'
        val = 1
        ax.plot(val, **style_dict[key])
        ax.set(**props)

        ax.grid(linestyle="--", alpha=0.2)
        ticks, labels = set_day_xtick(list(val), startDate, endDate)
        ax.xticks(ticks=ticks, labels=labels, rotation=30)
        plt.suptitle(gh, fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # ...

    #
    plt.savefig()
    plt.close()


def set_ytick(num, max_val, min_val):
    max_val = int(max_val)
    min_val = int(min_val)
    interval = (max_val - min_val) // (num-1)

    r, q = abs(interval) % 5, abs(interval)//5
    interval = (q+1)*5 if r > 1 else interval

    r, q = abs(min_val) % 5, abs(min_val)//5
    min_val = int(np.sign(min_val)) * (q+1)*5 if r > 0 else min_val

    ticks = list(range(min_val, max_val, interval))
    if len(ticks) < num:
        ticks.append(ticks[-1]+interval)

    return ticks


def set_day_xtick(num, var_list, startDate, endDate):

    interval = len(var_list) // (num-1) + 1
    ticks = range(0, len(var_list), interval)
    startDate_dt = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    endDate_dt = datetime.datetime.strptime(endDate, "%Y-%m-%d")

    labels = []
    ticks_ = []
    for i in ticks:
        cur_dt = startDate_dt + datetime.timedelta(days=i)
        cur_time = cur_dt.strftime("%m-%d")
        labels.append(cur_time)
        ticks_.append(i)
    labels.append(endDate_dt.strftime("%m-%d"))
    ticks_.append(len(var_list))

    return ticks_, labels


def set_hour_xtick(var_list, startDate, endDate):

    interval = len(var_list) // 6 + 1
    ticks = range(0, len(var_list), interval)
    startDate_dt = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    endDate_dt = datetime.datetime.strptime(endDate, "%Y-%m-%d")

    labels = []
    ticks_ = []
    for i in ticks:
        cur_dt = startDate_dt + datetime.timedelta(hours=i)
        cur_time = cur_dt.strftime("%m-%d")
        labels.append(cur_time)
        ticks_.append(i)
    labels.append(endDate_dt.strftime("%m-%d"))
    ticks_.append(len(var_list))

    return ticks_, labels
