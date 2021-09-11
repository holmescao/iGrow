import numpy as np
import random
import scipy.io as scio
import os
import datetime


def cal_days(startDate, endDate):
    st = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    et = datetime.datetime.strptime(endDate, "%Y-%m-%d")

    return (et - st).days + 1


def get_var_range(config, vars_dict, action_dims):
    bound0 = []
    bound1 = []
    for var, attr in vars_dict.items():
        low = config['action'][var]['min']
        up = config['action'][var]['max']
        bound0 += [low // attr[1]] * action_dims
        bound1 += [up // attr[1]] * action_dims

    return bound0, bound1


def get_expr_name(setting, args):
    params = list(setting['params_info'].keys())
    expr_name = ''
    for arg in params:
        argv = getattr(args, arg)
        expr_name += '_'+'@'.join([arg, str(argv)])

    experiment = 'global' + expr_name

    return experiment


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


def init_policy(config, vars_dict, plant_periods, X_best_sofar_path):
    if not os.path.exists(X_best_sofar_path):
        period_action = []
        for _ in range(plant_periods):
            for a, sets in vars_dict.items():
                a_desc = config['action'][a]
                min_a, max_a = int(a_desc['min']), int(a_desc['max'])
                action_list = [random.choice(range(min_a, max_a+1, sets[1]))
                               for _ in range(sets[0])]
                period_action += action_list

        scio.savemat(X_best_sofar_path, {'policy': period_action})
        print("generate init policy, saved in: %s" % X_best_sofar_path)
    else:
        print("%s already exists." % X_best_sofar_path)


def get_init_action(X, vars_dict, day_dims):
    X_np = np.array(X)
    X_np = X_np.reshape((len(X_np)//day_dims, day_dims), order='C')
    col_idx = 0
    x_init = []
    for _, sets in vars_dict.items():
        val = X_np[:, col_idx: col_idx+sets[2]]
        val = val.flatten() // sets[1]
        x_init += list(val)
        col_idx += sets[2]

    return x_init
