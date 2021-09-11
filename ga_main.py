from functools import wraps
from scipy import io as scio
import time
import argparse
import random
import sys
import scipy.io as scio
import numpy as np

import GA.ga_module.utils as utils
from GA.ga_module.custom import init_params
from GA.ga_module.method import blackbox_opt_mask, geatpy2_maximize_global_psy_v0
from GA.ga_module.Logger import Logger
from GA.ga_module.config import setting_train

from TenSim.simulator import PredictModel
from TenSim.config import config


def RunTime(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print("Run Time: %.2f min" % ((end_time-start_time)/60))
        return res

    return decorated


def ModelBase(period_action):
    ten_env.reset()

    dims = 96
    reward = 0
    day = 0
    done = False
    period_action = np.array(period_action)
    while not done:
        a = period_action[day*dims: (day+1)*dims]
        _, r, done, _ = ten_env.step(a)
        day += 1

        reward += float(r)

    return reward


@RunTime
def train_EGA(args_):
    "==========================Parameter initialization==============================="
    startDate, endDate = config['start_date'], config['end_date']
    plant_periods = utils.cal_days(startDate, endDate)
    vars_dict = setting_train['action_info']
    day_dims = sum(v[-1] for v in vars_dict.values())
    action_dims = plant_periods * day_dims // len(vars_dict.keys())
    bound = utils.get_var_range(config, vars_dict, action_dims)

    args = init_params(
        settings=setting_train['settings'],
        params=setting_train['params_info'],
        bound=bound,
        vars_dict=vars_dict,
        day_dims=day_dims,
        plant_periods=plant_periods,
        env=ModelBase,
        save_dir=args_.save_dir)

    experiment = utils.get_expr_name(setting=setting_train, args=args)

    utils.mkdir(args.X_best_sofar_path)
    utils.mkdir(args.log_path)

    args.X_best_sofar_path += '%s.mat' % experiment
    args.log_path += '%s.log' % experiment

    sys.stdout = Logger(args.log_path)
    sys.stderr = Logger(args.log_path)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    "==========================define policy==============================="
    utils.init_policy(config, vars_dict, plant_periods, args.X_best_sofar_path)

    X = list(scio.loadmat(args.X_best_sofar_path)['policy'][0])
    # temp+co2+illu+irri
    x_init = utils.get_init_action(
        X, vars_dict, day_dims)
    # init population
    x_inits = np.expand_dims(x_init, 0).repeat(args.NIND, axis=0)

    # mask
    x_mask = np.ones_like(x_inits[0])
    x_fixed = x_inits[0]

    # optimize
    x_best, _ = blackbox_opt_mask(opt=geatpy2_maximize_global_psy_v0, f=ModelBase,
                                  x_inits=x_inits, x_fixed=x_fixed, x_mask=x_mask,
                                  params=args)
    raw_policy = x_best[0]

    ''' Standardization policy'''
    policy = np.zeros_like(raw_policy)

    action_num = len(vars_dict.keys())
    raw_policy = raw_policy.reshape((action_num, -1), order='C')

    Idx = 0
    for d in range(plant_periods):
        for i, tup in enumerate(vars_dict.items()):
            dims = tup[1][2]
            multi = tup[1][1]
            policy[Idx:Idx+dims] = raw_policy[i, d*dims: (d+1)*dims] * multi

            Idx += dims

    # save the best policy
    scio.savemat(args.X_best_sofar_path, {
        'policy': list(policy)}, do_compression=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tmp_folder", default="result/", type=str)
    parser.add_argument("--save_dir", default="GA/", type=str)
    parser.add_argument("--version", default="baseline", type=str)
    args_ = parser.parse_args()

    """Instantiation simulator"""
    direcrory = args_.base_tmp_folder+'/models/'
    model_dir = direcrory + args_.version
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

    train_EGA(args_)
