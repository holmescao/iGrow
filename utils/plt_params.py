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
    "font.size": 18,     # 全局字号
    'font.family': 'STIXGeneral',  # 全局字体
    "figure.subplot.wspace": 0.2,  # 图-子图-宽度百分比
    "figure.subplot.hspace": 0.4,  # 图-子图-高度百分比
    "axes.spines.right": False,  # 坐标系-右侧线
    "axes.spines.top": False,   # 坐标系-上侧线
    "axes.titlesize": 22,   # 坐标系-标题-字号
    "axes.labelsize": 18,  # 坐标系-标签-字号
    "legend.fontsize": 18,  # 图例-字号
    'savefig.pad_inches': 0,  # 去除空白
    "xtick.labelsize": 16,  # 刻度-标签-字号
    "ytick.labelsize": 16,  # 刻度-标签-字号
    "xtick.direction": 'in',   # 刻度-方向
    "ytick.direction": 'in'  # 刻度-方向
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

    # 更新参数
    mpl.rcParams.update(params)
    props = {"xlabel": "days",
             "ylabel": "euro/m2"}

    # 基础类对象
    for i, ax in enumerate(axes.flat):
        key = 'AICU'
        val = 1
        ax.plot(val, **style_dict[key])  # 绘制曲线
        ax.set(**props)  # 参数设置

        # 美化
        ax.grid(linestyle="--", alpha=0.2)
        ticks, labels = set_day_xtick(list(val), startDate, endDate)
        ax.xticks(ticks=ticks, labels=labels, rotation=30)
        plt.suptitle(gh, fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # ...

    #
    plt.savefig()
    plt.close()


def set_day_xtick(num, var_list, startDate, endDate):
    # 设置坐标轴
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
    # 设置坐标轴
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
